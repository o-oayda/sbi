# Agent Notes

- When running under the Codex CLI environment there is no accessible GPU. Always set `JAX_PLATFORMS=cpu` for tests, scripts, and interactive runs; otherwise JAX will try to initialise CUDA and fail before doing any work.

## Python formatting

- Use Black for all Python files you create or modify. Read its settings from
  `pyproject.toml`: 88-character lines, preserved string quotes, and Python 3.12.
- Format only the files touched by the task; avoid unrelated formatting changes.
- Before finishing, run `uvx black --workers 1 <changed-python-files>` followed by
  `uvx black --check --workers 1 <changed-python-files>`. An existing Black
  installation may be used instead of `uvx`.
- Black is a formatter, not a substitute for code review or relevant tests.
  For formatting-only work, verify that the Python syntax trees are unchanged.
