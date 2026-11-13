# Contributing to FBQ

Thank you for your interest in contributing to FBQ! This document provides
guidelines for contributing to the project.

## Table of Contents

1. [Ways to Contribute](#ways-to-contribute)
2. [Development Setup](#development-setup)
3. [Code Standards](#code-standards)
4. [Pull Request Process](#pull-request-process)
5. [Reporting Issues](#reporting-issues)
6. [License](#license)

## Ways to Contribute

- **Bug reports**: File an issue with a clear description and reproduction steps
- **Feature requests**: Propose new features or enhancements
- **Code contributions**: Submit pull requests for bug fixes or new features
- **Documentation**: Improve documentation, examples, or tutorials
- **Testing**: Test on different platforms, report compatibility issues

## Development Setup

### Prerequisites

- Python 3.10–3.12
- `uv` package manager (recommended) or `pip`
- NVIDIA GPU with CUDA 12.6+ or 13.2+ (for GPU acceleration)
- FFmpeg (bundled by the launcher)

### Local Development

```bash
# Clone the repository
git clone https://github.com/your-org/FBQ.git
cd FBQ

# Install dependencies in editable mode
uv pip install -e . --config-settings editable_mode=compat

# Run the application
bash FBQ.sh
```

### Running Tests

```bash
# Syntax check all Python modules
python -m compileall main gui tools

# Run any unit tests (when available)
python -m pytest tests/ -v
```

## Code Standards

### Python Style

- Python 3.10–3.12 with `from __future__ import annotations`
- Type hints on public functions
- Use `dict[str, object]` payloads rather than dataclasses unless the module
  already uses dataclasses
- Follow existing code style in each module

### File Headers

Every source file must retain the copyright and SPDX header:

```python
# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
```

### Architecture Rules

1. **Config-driven**: No stage may read global state, environment paths, or
   hard-coded locations. Everything comes from `config.yaml`.
2. **Relative paths**: All config paths except `SESSION_PATH` are stored
   relative to `SESSION_PATH`. Use `to_relative_path` / `resolve_config_paths`.
3. **Determinism**: The master `RANDOM_SEED` is propagated to every child
   process. Any new randomness must derive from it.
4. **Fail fast**: A failed stage aborts the run; a failed LR pair aborts the
   sweep; a failed configuration aborts the batch queue.
5. **No telemetry**: Nothing may be uploaded. The only network calls are the
   GitHub release check and package-index access during setup.

### GUI Code

- Keep all Tk work on the main thread
- Long jobs run in a subprocess or worker thread
- Push updates through a queue that the Tk thread drains on a timer
- User-facing strings: plain English, sentence case, no trailing punctuation
  inside message boxes

## Pull Request Process

1. **Fork** the repository and create your branch from `main`
2. **Create an issue** first for significant changes (features, refactors)
3. **Make your changes** following the code standards above
4. **Test your changes**:
   - `python -m compileall main gui tools` passes
   - Application starts: `bash FBQ.sh` reaches the Home window
   - A full tracking run completes on a test video
5. **Update documentation** if needed (README, MANUAL.md, docstrings)
6. **Submit PR** with:
   - Clear description of changes
   - Link to related issue(s)
   - Screenshots for UI changes
7. **Address review feedback** promptly
8. **Maintainers will merge** after approval

### Commit Messages

Follow conventional commits format:

```
type(scope): brief description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(behavior): add pixel/cm calibration dialog

Add interactive calibration dialog for endpoints extraction.
Users can draw two points on a known distance and enter the
real-world measurement in cm.

Closes #123
```

## Reporting Issues

Before filing an issue:

1. Check existing issues for duplicates
2. Try to reproduce with the latest version
3. Gather relevant information

### Bug Reports

Include:
- FBQ version (`bash FBQ.sh --version` or Home window)
- Operating system and Python version
- GPU model and driver version (if applicable)
- Steps to reproduce
- Expected vs actual behavior
- Error messages / traceback
- Relevant log output (from the console window)

### Feature Requests

Include:
- Clear description of the proposed feature
- Use case / motivation
- Possible implementation approach
- Alternatives considered

## License

By contributing to FBQ, you agree that your contributions will be licensed
under the **GNU Affero General Public License v3.0 only** (AGPL-3.0-only),
the same license as the project.

All source files must retain the copyright and SPDX header:

```python
# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
```

### Attribution Requirements

FBQ is a derivative work of AMADEUS (Notomi, Matsumura & Dobata, 2026).
The upstream citation must be preserved in:
- README.md
- THIRD_PARTY_NOTICES.md
- AGENTS.md
- docs/MANUAL.md
- docs/REDESIGN.md

Never remove or modify the upstream attribution.

---

**Questions?** Open a [Discussion](https://github.com/your-org/FBQ/discussions) or file an issue.