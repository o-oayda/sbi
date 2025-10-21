from __future__ import annotations

import argparse
import os
import pathlib
import pty
import sys
import subprocess
from typing import List


def _waitstatus_to_exitcode(status: int) -> int:
    if hasattr(os, "waitstatus_to_exitcode"):
        return os.waitstatus_to_exitcode(status)
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        # Propagate signal as negative code, consistent with subprocess.
        return -os.WTERMSIG(status)
    return status


def run_with_logging(
    command: List[str],
    *,
    log_path: pathlib.Path,
    cwd: pathlib.Path | None = None,
    env_updates: dict[str, str] | None = None,
) -> int:
    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)

    original_cwd = pathlib.Path.cwd()
    if cwd is not None:
        os.chdir(cwd)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    original_env = os.environ.copy()
    os.environ.clear()
    os.environ.update(env)

    try:
        with open(log_path, "wb") as log_file:
            def master_read(fd: int) -> bytes:
                data = os.read(fd, 1024)
                if data:
                    log_file.write(data)
                    log_file.flush()
                return data

            try:
                status = pty.spawn(command, master_read=master_read)  # type: ignore[arg-type]
            except OSError:
                # Fallback: run without PTY while streaming output.
                log_file.flush()
                process = subprocess.Popen(
                    command,
                    cwd=os.getcwd(),
                    env=os.environ.copy(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                )
                assert process.stdout is not None
                for chunk in iter(lambda: process.stdout.readline(), b""):
                    if not chunk:
                        break
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    log_file.write(chunk)
                    log_file.flush()
                status = process.wait()
    finally:
        os.environ.clear()
        os.environ.update(original_env)
        os.chdir(original_cwd)

    return _waitstatus_to_exitcode(status)


def _parse_command_line(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a command with PTY logging.")
    parser.add_argument("--log", required=True, type=pathlib.Path, help="Path to write command output.")
    parser.add_argument("--cwd", type=pathlib.Path, help="Working directory for the command.")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable assignments to inject into the command environment.",
    )
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command to execute (precede with --).",
    )
    args = parser.parse_args(argv)
    if not args.cmd:
        parser.error("No command specified. Pass arguments after --.")
    if args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    if not args.cmd:
        parser.error("No command specified after --.")
    return args


def _main(argv: list[str] | None = None) -> int:
    args = _parse_command_line(argv)
    env_updates: dict[str, str] = {}
    for item in args.env:
        if "=" not in item:
            raise SystemExit(f"Invalid --env assignment '{item}', expected KEY=VALUE.")
        key, value = item.split("=", 1)
        env_updates[key] = value
    return run_with_logging(
        args.cmd,
        log_path=args.log,
        cwd=args.cwd,
        env_updates=env_updates,
    )


if __name__ == "__main__":
    raise SystemExit(_main())
