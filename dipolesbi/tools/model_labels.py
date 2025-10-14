from __future__ import annotations

from typing import Sequence

TOKEN_MAP = {
    "students-t": "Student's $t$",
    "studentst": "Student's $t$",
    "gauss": "Gaussian",
    "gaussian": "Gaussian",
    "cmb": "CMB",
    "direction": "direction",
}


def latexify_model(identifier: str) -> str:
    """
    Convert a model identifier string (e.g. ``free_students-t_extra_err``) into a
    label suitable for use in figure legends.
    """
    parts = [segment for segment in identifier.split("_") if segment]
    if parts and parts[0].lower() in {"free", "fixed"}:
        parts = parts[1:]

    suffix = ""
    if len(parts) >= 2 and [part.lower() for part in parts[-2:]] == ["extra", "err"]:
        parts = parts[:-2]
        suffix = " (extra error)"

    labels: list[str] = []
    for token in parts:
        key = token.lower()
        mapped = TOKEN_MAP.get(key)
        if mapped is not None:
            labels.append(mapped)
            continue
        cleaned = token.replace("-", " ")
        if cleaned.isupper():
            labels.append(cleaned)
        else:
            labels.append(" ".join(word.capitalize() for word in cleaned.split()))

    if not labels:
        base = identifier.replace("_", " ")
    else:
        base = " ".join(labels)

    label = (base + suffix).strip()
    return label or identifier


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Latexify model identifiers.")
    parser.add_argument("identifier", help="Model identifier string to format.")
    args = parser.parse_args(argv)

    print(latexify_model(args.identifier), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
