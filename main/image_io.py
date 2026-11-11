# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Checked writing of image files.

A write that fails under I/O pressure leaves a damaged file behind, and the
stage that reads it later fails with a message that points at the wrong
place. The stages that write many images therefore write to a temporary file
first, force the data to the disk and only then give the file its final name.
A reader never meets a half-written image, and a file that was damaged on the
way to the disk does not replace a good one.
"""

from __future__ import annotations

import os
import struct
import threading
import time
import zlib

import cv2

#: The signature every PNG file starts with.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunks_are_valid(path: str) -> bool:
    """Report whether every chunk of the PNG file matches its checksum.

    The check mirrors what the PNG decoder does, so a file whose data was
    damaged on the way to the disk is recognised here and not much later by
    the stage that reads it.
    """
    try:
        with open(path, "rb") as handle:
            if handle.read(len(_PNG_SIGNATURE)) != _PNG_SIGNATURE:
                return False
            while True:
                head = handle.read(8)
                if len(head) < 8:
                    return False
                length = struct.unpack(">I", head[:4])[0]
                chunk_type = head[4:8]
                data = handle.read(length)
                if len(data) != length:
                    return False
                checksum = handle.read(4)
                if len(checksum) != 4:
                    return False
                expected = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
                if struct.unpack(">I", checksum)[0] != expected:
                    return False
                if chunk_type == b"IEND":
                    return True
    except OSError:
        return False


def _force_to_disk(path: str) -> None:
    """Ask the drive to keep the data of the file, not only the page cache."""
    try:
        handle = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


def _force_folder_to_disk(folder: str) -> None:
    """Ask the drive to keep the name of the file, not only the page cache."""
    try:
        handle = os.open(folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


def _temporary_path(path: str) -> str:
    """Return a name beside the image that no other writer can pick."""
    root, extension = os.path.splitext(path)
    return f"{root}.part-{os.getpid()}-{threading.get_ident()}{extension}"


def write_png(path: str, image, *, params=None, attempts: int = 3) -> str:
    """Write a PNG file and confirm that it is complete and readable.

    Returns the path of the written file. The image is written to a
    temporary file, checked, forced to the disk and then moved onto the
    wanted name. A write that fails or leaves a damaged file is retried, and
    a file that stays damaged aborts the stage with the path of the file that
    could not be written.
    """
    reason = "the image could not be written"
    folder = os.path.dirname(os.path.abspath(path))
    for attempt in range(max(1, attempts)):
        temporary = _temporary_path(path)
        try:
            written = bool(
                cv2.imwrite(temporary, image, params)
                if params
                else cv2.imwrite(temporary, image)
            )
            if written and _png_chunks_are_valid(temporary):
                _force_to_disk(temporary)
                os.replace(temporary, path)
                _force_folder_to_disk(folder)
                return path
            reason = "the written file was damaged" if written else reason
        except Exception as exc:  # the codec reports failures in several ways
            reason = str(exc)
        finally:
            if os.path.exists(temporary):
                try:
                    os.remove(temporary)
                except OSError:
                    pass
        time.sleep(0.2 * (attempt + 1))
    raise RuntimeError(f"Failed to write the image {path}: {reason}")


def read_png(path: str):
    """Read an image and report a damaged file with a clear message."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read the image {path}")
    return image
