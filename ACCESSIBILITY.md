# Accessibility Statement

## Commitment

FBQ is committed to making its software accessible to the widest possible
audience, including researchers with disabilities. We strive to follow
accessibility best practices and continuously improve the user experience.

## Supported Environments

| Platform | GUI Framework | Accessibility Support |
|----------|---------------|----------------------|
| Windows 10/11 | CustomTkinter (Tk) | Native Windows accessibility APIs |
| macOS (Apple Silicon) | CustomTkinter (Tk) | VoiceOver compatible |
| Linux / WSL2 | CustomTkinter (Tk) | AT-SPI / Orca compatible |

## Current Accessibility Features

### Visual
- **High contrast dark theme** (default): Meets WCAG AA contrast ratios
- **Scalable UI**: All elements respect system DPI scaling
- **Focus indicators**: Visible focus rings on all interactive elements
- **Text sizing**: Configurable font sizes in display settings

### Keyboard Navigation
- **Tab order**: Logical tab sequence through all controls
- **Keyboard shortcuts**: Standard Tk shortcuts (Alt+key for menus)
- **Focus management**: Focus returns to logical position after dialogs

### Screen Readers
- **Semantic labels**: All controls have accessible names
- **Live regions**: Progress updates announced via ARIA live regions
- **State announcements**: Checkbox, radio, combobox states announced

### Motor
- **No drag-only operations**: All canvas interactions have keyboard alternatives
- **Adjustable click targets**: Minimum 44×44px touch targets
- **Timeout controls**: No auto-advancing without user action

## Known Limitations

| Area | Limitation | Workaround |
|------|------------|------------|
| ROI drawing | Requires mouse for polygon creation | Pre-defined rectangle/circle ROI shapes available |
| Video playback | Canvas-based, limited screen reader support | Export CSV for data analysis; use external players |
| Color-coded tracks | Color-only identification | Track IDs shown as text labels; configurable |
| Real-time plots | Matplotlib canvas not fully accessible | Export kinetics CSV for accessible analysis |

## Configuration for Accessibility

### Display Settings (Behavior Quantification dialog)
- **Arrow size**: Reduce for less visual clutter
- **ID font size**: Increase for readability
- **Box thickness**: Increase for better visibility
- **Trail length**: Reduce to minimize motion

### System-Level
- **High contrast mode**: Enable OS high contrast; FBQ respects system colors
- **Large text**: Enable OS text scaling; FBQ UI scales accordingly
- **Reduce motion**: Enable OS reduce motion; trail animations respect this

## Reporting Barriers

If you encounter an accessibility barrier, please report it:

1. **GitHub Issue**: Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.yml) with label `accessibility`
2. **Email**: fbq-accessibility@users.noreply.github.com

Include:
- Description of the barrier
- Assistive technology used (screen reader, magnification, voice control, etc.)
- Steps to reproduce
- Suggested improvement (if any)

## Testing

We test accessibility with:
- NVDA / JAWS (Windows)
- VoiceOver (macOS)
- Orca (Linux)
- Windows High Contrast Mode
- macOS Dark Mode / Increase Contrast
- Keyboard-only navigation

## Future Improvements

Planned accessibility enhancements:
- [ ] Explicit keyboard shortcuts for all major actions
- [ ] Screen reader announcements for tracking progress
- [ ] Accessible data table view for CSV results
- [ ] Configurable color palette for track identification
- [ ] Audio cues for long-running operations

## Conformance

This statement covers FBQ v1.2.4 and later. We aim for **WCAG 2.1 Level AA**
conformance for the GUI application. The command-line pipeline stages are
designed for scriptable/automated use and follow CLI accessibility conventions.

---

**Last updated**: 2026-11-13
**Version**: 1.0