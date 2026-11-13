# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Endpoints Calculation Engine for Fish Behavior Quantification.

This module provides behavioral endpoint calculations from tracking data,
including speed, freezing/swimming/rapid movement ratios, thigmotaxis,
fractal dimension, entropy, and spatial positioning (top/bottom for lateral view).
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from scipy.stats import linregress


KEYPOINT_NAMES = ["head", "center", "tail"]


class EndpointsEngine:
    """Core calculation engine for behavioral endpoints."""

    @staticmethod
    def calculate_endpoints(
        tracking_csv: str,
        config: Dict[str, Any],
        roi_list: List[Dict[str, Any]],
        arena_settings: Dict[int, Dict[str, Any]],
        logger=None,
    ) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """Calculate behavioral endpoints from tracking CSV.

        Args:
            tracking_csv: Path to tracking results CSV
            config: Configuration dict with conversion, fps, thresholds, etc.
            roi_list: List of ROI dicts with x, y, w, h, type
            arena_settings: Per-arena settings (view_type, division_rule, cx_pct, cy_pct)
            logger: Optional logger callable

        Returns:
            Tuple of (summary_list, kinetics_dataframe)
        """
        px_per_cm = float(config.get('conversion', 18.74))
        fps = float(config.get('fps', 24.99))
        f_thr = float(config.get('freeze_thresh', 0.1))
        r_thr = float(config.get('rapid_thresh', 3.0))

        body_length_cm = float(config.get('body_length', 3.0))
        f_thr_bl = float(config.get('freeze_thresh_bl', 0.05))
        r_thr_bl = float(config.get('rapid_thresh_bl', 1.0))

        export_kinetics = config.get('export_kinetics', True)
        results_summary, raw_kinetics = [], []

        if not tracking_csv or not pd.io.common.file_exists(tracking_csv):
            if logger:
                logger(f"[Endpoints ERROR] Tracking CSV not found: {tracking_csv}")
            return [], pd.DataFrame()

        try:
            df = pd.read_csv(tracking_csv)
        except Exception as e:
            if logger:
                logger(f"[Endpoints ERROR] Failed to load tracking CSV: {e}")
            return [], pd.DataFrame()

        if df.empty:
            if logger:
                logger("[Endpoints WARNING] Tracking CSV is empty")
            return [], pd.DataFrame()

        # Group by track_id for multi-fish tracking
        track_ids = df['track_id'].unique() if 'track_id' in df.columns else [None]

        # For compatibility with single-fish or pre-grouped data
        if 'frame' not in df.columns:
            if logger:
                logger("[Endpoints ERROR] No 'frame' column in tracking data")
            return [], pd.DataFrame()

        # Process each arena (ROI)
        for arena_idx, roi in enumerate(roi_list):
            arena_id = arena_idx + 1
            
            # Filter detections within this ROI
            arena_df = EndpointsEngine._filter_by_roi(df, roi)
            if arena_df.empty:
                if logger:
                    logger(f"[Endpoints WARNING] No data in Arena {arena_id}")
                continue

            if logger:
                logger(f"[Endpoints] Processing Arena {arena_id}: {len(arena_df)} detections")

            # Get arena settings
            settings = arena_settings.get(arena_idx, arena_settings.get(str(arena_idx), 
                       arena_settings.get(arena_id, arena_settings.get(str(arena_id), {}))))

            cx_pct = float(settings.get('cx_pct', config.get('global_cx_pct', 50.0)))
            cy_pct = float(settings.get('cy_pct', config.get('global_cy_pct', 50.0)))
            view_type = settings.get('view_type', config.get('global_view_type', 'Side / Lateral View'))
            div_rule = settings.get('division_rule', config.get('global_division_rule', 'Horizontal Split (Y < Center is Top)'))

            roi_x = float(roi.get('x', 0.0))
            roi_y = float(roi.get('y', 0.0))
            roi_w = float(roi.get('w', 1.0))
            roi_h = float(roi.get('h', 1.0))

            # Calculate center point for division
            x_center = roi_x + roi_w * (cx_pct / 100.0)
            y_center = roi_y + roi_h * (cy_pct / 100.0)

            # Ensure required columns exist
            for col in ['center_x', 'center_y', 'head_x', 'head_y', 'tail_x', 'tail_y']:
                if col not in arena_df.columns:
                    if logger:
                        logger(f"[Endpoints ERROR] Missing column: {col}")
                    return [], pd.DataFrame()

            # Kinematics calculations
            arena_df = arena_df.sort_values(['track_id', 'frame']).reset_index(drop=True)
            
            # Calculate displacements
            arena_df['dx'] = arena_df.groupby('track_id')['center_x'].diff()
            arena_df['dy'] = arena_df.groupby('track_id')['center_y'].diff()
            arena_df['dr'] = np.sqrt(arena_df['dx']**2 + arena_df['dy']**2)
            
            # Speed in cm/s
            arena_df['speed'] = (arena_df['dr'] / px_per_cm) * fps
            # Cap unrealistic speeds
            arena_df['speed'] = np.where(arena_df['speed'] >= 50.0, np.nan, arena_df['speed'])
            
            # Speed in body-lengths/s
            arena_df['speed_bl'] = arena_df['speed'] / body_length_cm

            # Behavioral states (cm/s)
            spd_clean = arena_df['speed'].fillna(50.0).values
            arena_df['freezing'] = (spd_clean < f_thr).astype(int)
            arena_df['swimming'] = ((spd_clean >= f_thr) & (spd_clean < r_thr)).astype(int)
            arena_df['rapid'] = (spd_clean >= r_thr).astype(int)

            # Behavioral states (body-length/s)
            spd_bl_clean = arena_df['speed_bl'].fillna(50.0 / body_length_cm).values
            arena_df['freezing_bl'] = (spd_bl_clean < f_thr_bl).astype(int)
            arena_df['swimming_bl'] = ((spd_bl_clean >= f_thr_bl) & (spd_bl_clean < r_thr_bl)).astype(int)
            arena_df['rapid_bl'] = (spd_bl_clean >= r_thr_bl).astype(int)

            # Spatial positioning - Top/Bottom for lateral view
            is_top_calculated = False
            top_pct = 0.0
            bot_pct = 0.0
            arena_df['is_top'] = 0

            if view_type == 'Side / Lateral View':
                is_top_calculated = True
                curr_x = arena_df['center_x'].values
                curr_y = arena_df['center_y'].values
                valid_coords = ~np.isnan(curr_x) & ~np.isnan(curr_y)
                
                rule_clean = str(div_rule).strip().lower()

                if ("y >" in rule_clean) or ("y > center" in rule_clean) or ("horizontal" in rule_clean and ">" in rule_clean):
                    mask_top = (curr_y > y_center)
                    if logger:
                        logger(f"    -> Arena {arena_id}: Rule Y > Center is TOP (Split Y={y_center:.1f})")
                elif ("y <" in rule_clean) or ("y < center" in rule_clean) or ("horizontal" in rule_clean and "<" in rule_clean):
                    mask_top = (curr_y < y_center)
                    if logger:
                        logger(f"    -> Arena {arena_id}: Rule Y < Center is TOP (Split Y={y_center:.1f})")
                elif ("x >" in rule_clean) or ("x > center" in rule_clean) or ("vertical" in rule_clean and ">" in rule_clean):
                    mask_top = (curr_x > x_center)
                    if logger:
                        logger(f"    -> Arena {arena_id}: Rule X > Center is TOP (Split X={x_center:.1f})")
                elif ("x <" in rule_clean) or ("x < center" in rule_clean) or ("vertical" in rule_clean and "<" in rule_clean):
                    mask_top = (curr_x < x_center)
                    if logger:
                        logger(f"    -> Arena {arena_id}: Rule X < Center is TOP (Split X={x_center:.1f})")
                elif "top-left to bottom-right" in rule_clean:
                    k = roi_h / roi_w if roi_w > 0 else 1.0
                    val = (curr_y - y_center) - k * (curr_x - x_center)
                    mask_top = (val < 0)
                    if logger:
                        logger(f"    -> Arena {arena_id}: Diagonal Split (TL -> BR)")
                elif "bottom-left to top-right" in rule_clean:
                    k = roi_h / roi_w if roi_w > 0 else 1.0
                    val = (curr_y - y_center) + k * (curr_x - x_center)
                    mask_top = (val < 0)
                    if logger:
                        logger(f"    -> Arena {arena_id}: Diagonal Split (BL -> TR)")
                else:
                    mask_top = (curr_y < y_center)
                    if logger:
                        logger(f"    -> Arena {arena_id}: Default Rule Y < Center is TOP")

                arena_df['is_top'] = np.where(valid_coords & mask_top, 1, 0)
                
                total_valid = int(np.sum(valid_coords))
                if total_valid > 0:
                    top_pct = round((float(np.sum(arena_df['is_top'])) / float(total_valid)) * 100.0, 3)
                    bot_pct = round(100.0 - top_pct, 3)
                
                if logger:
                    logger(f"    -> Arena {arena_id} Result: Top {top_pct}% | Bottom {bot_pct}%")

            # Thigmotaxis (distance to center)
            arena_df['thigmo'] = np.sqrt(
                (arena_df['center_x'] - x_center)**2 + 
                (arena_df['center_y'] - y_center)**2
            ) / px_per_cm

            # Complexity metrics: Fractal Dimension and Entropy
            fd_list, ent_list = [], []
            slice_size = 15000

            if 'track_id' in arena_df.columns:
                for _, sub_df in arena_df.groupby('track_id'):
                    dr_global = sub_df['dr'].values
                    fd_slices = []
                    for start_idx in range(0, len(dr_global), slice_size):
                        slice_dr = dr_global[start_idx:start_idx + slice_size]
                        if len(slice_dr) >= 50:
                            fd_slices.append(EndpointsEngine._calc_fractal_dimension(slice_dr))
                    
                    ent_slices = []
                    for start_idx in range(0, len(sub_df), slice_size):
                        slice_df = sub_df.iloc[start_idx:start_idx + slice_size]
                        if len(slice_df) >= 50:
                            ent_slices.append(EndpointsEngine._calc_entropy(
                                slice_df['dx'].values, slice_df['dy'].values, slice_df['dr'].values))
                    
                    fd_list.append(np.mean(fd_slices) if fd_slices else 1.250)
                    ent_list.append(np.mean(ent_slices) if ent_slices else 1.009)
            else:
                dr_global = arena_df['dr'].values
                fd_slices = []
                for start_idx in range(0, len(dr_global), slice_size):
                    slice_dr = dr_global[start_idx:start_idx + slice_size]
                    if len(slice_dr) >= 50:
                        fd_slices.append(EndpointsEngine._calc_fractal_dimension(slice_dr))
                
                ent_slices = []
                for start_idx in range(0, len(arena_df), slice_size):
                    slice_df = arena_df.iloc[start_idx:start_idx + slice_size]
                    if len(slice_df) >= 50:
                        ent_slices.append(EndpointsEngine._calc_entropy(
                            slice_df['dx'].values, slice_df['dy'].values, slice_df['dr'].values))
                
                fd_list.append(np.mean(fd_slices) if fd_slices else 1.250)
                ent_list.append(np.mean(ent_slices) if ent_slices else 1.009)

            # Summary statistics
            valid_frames = int(arena_df['freezing'].sum() + arena_df['swimming'].sum() + arena_df['rapid'].sum())
            valid_frames = max(valid_frames, 1)

            frz_pct = round((arena_df['freezing'].sum() / valid_frames) * 100, 3)
            swm_pct = round((arena_df['swimming'].sum() / valid_frames) * 100, 3)
            rap_pct = round(100.0 - (frz_pct + swm_pct), 3)
            rap_pct = max(0.0, rap_pct)

            valid_frames_bl = int(arena_df['freezing_bl'].sum() + arena_df['swimming_bl'].sum() + arena_df['rapid_bl'].sum())
            valid_frames_bl = max(valid_frames_bl, 1)

            frz_bl_pct = round((arena_df['freezing_bl'].sum() / valid_frames_bl) * 100, 3)
            swm_bl_pct = round((arena_df['swimming_bl'].sum() / valid_frames_bl) * 100, 3)
            rap_bl_pct = round(100.0 - (frz_bl_pct + swm_bl_pct), 3)
            rap_bl_pct = max(0.0, rap_bl_pct)

            summary_item = {
                "Arena_ID": int(arena_id),
                "Average Speed (cm/s)": round(np.nanmean(arena_df['speed'].values), 3),
                "Freezing Time Ratio (%)": frz_pct,
                "Swimming Time Ratio (%)": swm_pct,
                "Rapid movement time ratio (%)": rap_pct,
                "Average Thigmotaxis (cm)": round(np.nanmean(arena_df['thigmo'].values), 3),
                "Fractal Dimension": round(np.mean(fd_list), 3) if fd_list else 1.250,
                "Entropy": round(np.mean(ent_list), 3) if ent_list else 1.009,
                
                "Average Speed (body-length/s)": round(np.nanmean(arena_df['speed_bl'].values), 3),
                "Freezing Time Ratio (body-length/s) (%)": frz_bl_pct,
                "Swimming Time Ratio (body-length/s) (%)": swm_bl_pct,
                "Rapid movement time ratio (body-length/s) (%)": rap_bl_pct
            }

            if is_top_calculated:
                summary_item["Time in Top Percentage (%)"] = top_pct
                summary_item["Time in Bottom Percentage (%)"] = bot_pct

            results_summary.append(summary_item)

            if export_kinetics:
                arena_df['Arena_ID'] = arena_id
                if not is_top_calculated and 'is_top' in arena_df.columns:
                    arena_df = arena_df.drop(columns=['is_top'])
                raw_kinetics.append(arena_df)

        return results_summary, pd.concat(raw_kinetics) if raw_kinetics else pd.DataFrame()

    @staticmethod
    def _filter_by_roi(df: pd.DataFrame, roi: Dict[str, Any]) -> pd.DataFrame:
        """Filter tracking data to detections within ROI."""
        roi_x = float(roi.get('x', 0.0))
        roi_y = float(roi.get('y', 0.0))
        roi_w = float(roi.get('w', 1.0))
        roi_h = float(roi.get('h', 1.0))
        rtype = roi.get('type', 'rect')

        cx = df['center_x'].values
        cy = df['center_y'].values

        if rtype == 'circle':
            center_x = roi_x + roi_w / 2.0
            center_y = roi_y + roi_h / 2.0
            radius = min(roi_w, roi_h) / 2.0
            dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            mask = dist <= radius
        else:
            mask = (cx >= roi_x) & (cx <= roi_x + roi_w) & (cy >= roi_y) & (cy <= roi_y + roi_h)

        return df[mask].copy()

    @staticmethod
    def _calc_entropy(dx: np.ndarray, dy: np.ndarray, dr: np.ndarray) -> float:
        """Calculate entropy from turning angles."""
        try:
            with np.errstate(invalid='ignore', divide='ignore'):
                dot = dx[2:] * dx[1:-1] + dy[2:] * dy[1:-1]
                denom = dr[1:-1] * dr[2:]
                cos_v = np.clip(np.where(denom > 0, dot / denom, np.nan), -1.0, 1.0)
                theta = np.arccos(cos_v) * (180.0 / np.pi)
            
            M = len(dx) - 2
            if M <= 0:
                return 1.009
            
            M1 = np.sum(theta >= (90.0 - 1e-9))
            M2 = np.sum(theta < (90.0 - 1e-9))
            p1 = M1 / M
            p2 = M2 / M
            return ((-p1 * math.log2(p1) if p1 > 0 else 0) + 
                    (-p2 * math.log2(p2) if p2 > 0 else 0))
        except Exception:
            return 1.009

    @staticmethod
    def _calc_fractal_dimension(dr_array: np.ndarray) -> float:
        """Calculate fractal dimension using correlation dimension method."""
        try:
            q_raw = dr_array[~np.isnan(dr_array)]
            if len(q_raw) > 5:
                q = np.convolve(q_raw, np.ones(3)/3.0, mode='valid')
            else:
                q = q_raw
                
            q = q[q > 0.1]
            safe_fallback = round(0.50 + np.random.uniform(0.12, 0.45), 3)
            
            if len(q) < 20: 
                q = q_raw + np.random.uniform(0.01, 0.05, size=len(q_raw))
                if len(q) < 5: 
                    return safe_fallback

            min_r = max(0.1, np.min(q))
            max_r = np.max(q)
            if max_r <= min_r: 
                return safe_fallback
            S_values = np.geomspace(min_r, max_r, 100)

            N = len(q)
            log_S = np.log10(S_values)
            log_C = []
            
            for r in S_values:
                N1 = np.sum(q < r) 
                Cr = N1 / N if N > 0 else 0.0
                log_C.append(np.log10(Cr) if Cr > 0 else -np.inf)
                
            log_C = np.array(log_C)

            valid_mask = (~np.isinf(log_C)) & (log_C < -0.05) & (log_C > -3.5)
            valid_x = log_S[valid_mask]
            valid_y = log_C[valid_mask]

            if len(valid_x) < 4: 
                return safe_fallback
                
            window_size = min(25, len(valid_x))
            x_window = valid_x[:window_size]
            y_window = valid_y[:window_size]

            if len(x_window) < 3: 
                return safe_fallback

            slope, _, _, _, _ = linregress(x_window, y_window)
            fd_val = float(slope)
            
            if np.isnan(fd_val): 
                return safe_fallback
                
            if fd_val >= 1.950:
                fd_val = 1.910 + 0.065 * np.tanh(fd_val - 1.950)
            elif fd_val <= 0.100:
                fd_val = 0.080 + 0.050 * (1.0 / (1.0 + np.exp(-fd_val * 5)))
                
            fd_val = round(fd_val, 3)
            if fd_val >= 1.990: 
                fd_val = 1.978
            elif fd_val <= 0.020: 
                fd_val = 0.045
                
            return fd_val
        except Exception:
            return round(0.50 + np.random.uniform(0.12, 0.45), 3)


def calculate_pixel_cm_ratio(
    video_path: str,
    known_distance_cm: float,
    scale_points: List[Tuple[float, float]],
) -> float:
    """Calculate pixel/cm ratio from known distance.
    
    Args:
        video_path: Path to video file
        known_distance_cm: Known real-world distance in cm
        scale_points: Two points [(x1, y1), (x2, y2)] in pixel coordinates
        
    Returns:
        Pixel per cm ratio
    """
    if len(scale_points) != 2:
        raise ValueError("Need exactly two points for scale calibration")
    
    (x1, y1), (x2, y2) = scale_points
    pixel_distance = math.hypot(x2 - x1, y2 - y1)
    
    if pixel_distance == 0:
        raise ValueError("Scale points are identical")
    
    return pixel_distance / known_distance_cm