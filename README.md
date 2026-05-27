# Janus Process

## Local development

This repository uses a local Python virtual environment for development and testing.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e '.[dev]'
```

### Run tests

```bash
python -m pytest -q
```

### Notes

- The `.venv/` directory is excluded from git via `.gitignore`.
- Use the virtual environment for all local Python work to avoid interfering with system packages.
