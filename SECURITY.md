# Security Policy

## Supported Versions

FBQ follows a rolling release model. Security fixes are applied to the latest
release. We recommend always using the most recent version.

| Version | Supported |
| ------- | --------- |
| Latest release | ✅ |
| Older releases | ❌ |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to **fbq-security@users.noreply.github.com**.

Include the following information:

1. **Description** of the vulnerability
2. **Steps to reproduce** or proof-of-concept
3. **Impact** assessment (what could an attacker achieve?)
4. **Affected versions** (if known)
5. **Suggested fix** (if any)

We will acknowledge receipt within 48 hours and provide a status update within
5 business days.

## Security Considerations

### Network Access

FBQ is designed to run **offline** during tracking operations. The only network
access occurs during:

1. **Initial setup**: Package installation via `uv` (PyPI, PyTorch index)
2. **GitHub release check**: A 2-second check for updates on startup
3. **Colab packaging**: Optional Google Drive upload (explicit user action)

No telemetry, usage data, or video data is ever uploaded.

### File System Access

- FBQ reads video files from user-specified directories
- FBQ writes results to user-specified output directories
- FBQ creates a virtual environment in `~/.fbq/venv/` (configurable via
  `FBQ_VENV`)
- FBQ stores settings in `~/.config/fbq/` (standard XDG config location)

### Dependencies

- Dependencies are pinned in `pyproject.toml` and `uv.lock`
- PyTorch wheels are fetched from official PyTorch index
- FFmpeg is a pinned, checksum-verified static build from BTBN
- All dependencies are installed into an isolated virtual environment

### Model Weights

- Model weights (`.pt` files) are loaded from the user's trained session
- No pre-trained weights are downloaded automatically
- YOLO11n-OBB weights bundled for initial testing only (`yolo11n-obb.pt`)

### AGPL-3.0 Compliance

FBQ is licensed under AGPL-3.0-only. If you deploy a modified version as a
network service, you must provide the corresponding source code to users
interacting with it over the network (AGPL-3.0 §5).

The Colab runtime package includes a per-file SHA-256 manifest recording
exactly which source was used — keep that mechanism intact.

## Disclosure Policy

When a security vulnerability is confirmed:

1. We will develop a fix in a private fork
2. We will release a patched version
3. We will publish a security advisory on GitHub
4. We will credit the reporter (unless they request anonymity)

## Scope

This policy covers the FBQ codebase in this repository. It does not cover:

- Upstream AMADEUS code (see their security policy)
- Third-party dependencies (report to their maintainers)
- User environment issues (OS, drivers, hardware)

---

**Questions?** Contact **fbq-security@users.noreply.github.com**