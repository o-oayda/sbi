# Agent Notes

- When running under the Codex CLI environment there is no accessible GPU. Always set `JAX_PLATFORMS=cpu` for tests, scripts, and interactive runs; otherwise JAX will try to initialise CUDA and fail before doing any work.
