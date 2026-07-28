"""Copy a verified analysis artifact into a destination repository cache."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_digest(checksum_path: Path, archive_name: str) -> str:
    fields = checksum_path.read_text(encoding="utf-8").split()
    if len(fields) != 2 or fields[1] != archive_name:
        raise ValueError(f"Invalid checksum file: {checksum_path}")
    digest = fields[0].lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"Invalid SHA-256 digest in: {checksum_path}")
    return digest


def copy_artifact(
    experiment_id: str,
    destination_repository: Path,
    artifacts_root: Path,
) -> Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment_id) is None:
        raise ValueError(f"Invalid experiment ID: {experiment_id!r}")
    source_dir = artifacts_root / experiment_id
    archive = source_dir / f"{experiment_id}.analysis.zip"
    checksum = source_dir / f"{experiment_id}.analysis.zip.sha256"
    if not archive.is_file() or not checksum.is_file():
        raise FileNotFoundError(
            f"Expected packaged artifact and checksum below: {source_dir}"
        )

    expected = expected_digest(checksum, archive.name)
    actual = sha256(archive)
    if actual != expected:
        raise ValueError(
            f"Artifact checksum mismatch: expected {expected}, calculated {actual}"
        )

    repository = destination_repository.expanduser().resolve(strict=True)
    if not (repository / ".git").exists():
        raise ValueError(f"Destination path is not a Git repository: {repository}")

    cache = repository / "artifacts" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / f"{experiment_id}.{actual}.analysis.zip"
    if destination.exists():
        if sha256(destination) != actual:
            raise ValueError(
                f"Existing cache entry has the wrong contents: {destination}"
            )
        return destination

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        shutil.copyfile(archive, temporary)
        if sha256(temporary) != actual:
            raise ValueError("Copied artifact failed checksum verification")
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    parser.add_argument("destination_repository", type=Path)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    try:
        destination = copy_artifact(
            args.experiment_id,
            args.destination_repository,
            args.artifacts_root,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from None

    digest = sha256(destination)
    print(f"Copied artifact: {destination}")
    print(f"SHA-256: {digest}")
    print("The destination's trusted artifact registry was not modified.")


if __name__ == "__main__":
    main()
