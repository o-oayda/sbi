from __future__ import annotations

import argparse
import math
import pathlib
import re
import sys
from typing import Iterable, List, Sequence, Tuple

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
RICH_TAG = re.compile(r"\[[^\]]+\]")


def strip_formatting(text: str) -> str:
    """Remove Rich/ANSI formatting and surrounding whitespace."""
    text = ANSI_ESCAPE.sub("", text)
    text = RICH_TAG.sub("", text)
    return text.strip()


def parse_logz_from_log(file_path: pathlib.Path) -> Tuple[float, float]:
    """
    Extract the mean and standard deviation from the CLI log output.
    """
    target_line: str | None = None
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        clean = strip_formatting(raw_line)
        if not clean:
            continue
        lower = clean.lower()
        if "average logz (" in lower or "bootstrap average logz (" in lower:
            if "±" in clean:
                target_line = clean

    if target_line is None:
        raise ValueError("Average logZ line not found in log output.")

    try:
        _, value_part = target_line.rsplit(":", 1)
    except ValueError as exc:
        raise ValueError("Unable to split logZ line on ':'") from exc

    value_part = strip_formatting(value_part)

    if "±" not in value_part:
        raise ValueError("LogZ line is missing '±' separator.")

    mean_str, std_str = (segment.strip() for segment in value_part.split("±", 1))
    return float(mean_str), float(std_str)


def write_tables(
    entries: Sequence[Tuple[str, str, float, float, str]],
    output_dir: pathlib.Path,
) -> None:
    """
    Generate Markdown and LaTeX tables summarising logZ values.
    """
    sorted_entries = sorted(entries, key=lambda item: item[2], reverse=True)

    md_lines = ["| Run | Model | $\\ln \\mathcal{Z}$ |", "| --- | --- | --- |"]
    for run_name, label, mean, std, _ in sorted_entries:
        md_lines.append(f"| {run_name} | {label} | ${mean:.1f} \\pm {std:.1f}$ |")

    def latex_escape(text: str) -> str:
        replacements = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
            "\\": r"\textbackslash{}",
        }
        return "".join(replacements.get(char, char) for char in text)

    tex_lines = [
        r"\begin{tabular}{lll}",
        r"\hline",
        r"Run & Model & $\ln \mathcal{Z}$ \\",
        r"\hline",
    ]
    for run_name, label, mean, std, _ in sorted_entries:
        run_tex = latex_escape(run_name)
        label_tex = label  # label may contain LaTeX already
        tex_lines.append(f"{run_tex} & {label_tex} & \\num{{{mean:.1f} +- {std:.1f}}} \\\\")
    tex_lines.extend([r"\hline", r"\end{tabular}"])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logz_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (output_dir / "logz_summary.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    if sorted_entries:
        best_mean = sorted_entries[0][2]
        best_std = sorted_entries[0][3]

        md_b_lines = ["| Model | $\\Delta \\ln \\mathcal{Z}$ |", "| --- | --- |"]
        tex_b_lines = [
            r"\begin{tabular}{l r@{\,\,\pm\,\,}l}",
            r"\hline",
            r"Model & \multicolumn{2}{c}{$\ln B$} \\",
            r"\hline",
        ]

        CUSTOM_LNB_LABELS = {
            "cmb_direction": (
                "CMB direction, free velocity", 
                "CMB direction, free velocity"
            ),
            "cmb_velocity": (
                "CMB velocity, free direction", 
                "CMB velocity, free direction"
            ),
            "cmb_dipole": (
                "CMB velocity & direction", 
                "CMB velocity \\& direction"
            ),
            "free_gauss": (
                "Free dipole, no extra error, Gaussian", 
                "Free dipole, no extra error, Gaussian"
            ),
            "free_gauss_extra_err": (
                "Free dipole, extra error, Gaussian", 
                "Free dipole, extra error, Gaussian"
            ),
            "free_students-t_extra_err": (
                "Free dipole, extra error, Student's $t$",
                "Free dipole, extra error, Student's $t$",
            ),
            "free_students-t": (
                "Free dipole, no extra error, Student's $t$",
                "Free dipole, no extra error, Student's $t$",
            ),
        }
        SKIP_IDENTIFIERS = {"secrest2021", "dam2023"}
        logged_skips: set[str] = set()

        def resolve_bayes_label(label: str, identifier: str) -> tuple[str, str]:
            if identifier in CUSTOM_LNB_LABELS:
                return CUSTOM_LNB_LABELS[identifier]
            if identifier in SKIP_IDENTIFIERS and identifier not in logged_skips:
                print(
                    f"[logz_summary] Skipping custom label for Bayes factor table: '{identifier}' not yet available.",
                    file=sys.stderr,
                )
                logged_skips.add(identifier)
            return label, label

        for idx, (run_name, label, mean, std, identifier) in enumerate(sorted_entries):
            delta_mean = mean - best_mean
            delta_std = math.sqrt(std * std + best_std * best_std)
            display_label_md, display_label_tex = resolve_bayes_label(label, identifier)
            if idx == 0:
                md_b_lines.append(f"| {display_label_md} | $0.0$ |")
                tex_b_lines.append(f"{display_label_tex} & $0$ & $0$ \\\\")
            else:
                md_b_lines.append(
                    f"| {display_label_md} | ${delta_mean:.1f} \\pm {delta_std:.1f}$ |"
                )
                tex_b_lines.append(
                    f"{display_label_tex} & ${delta_mean:.1f}$ & ${delta_std:.1f}$ \\\\"
                )

        tex_b_lines.extend([r"\hline", r"\end{tabular}"])

        (output_dir / "logb_summary.md").write_text("\n".join(md_b_lines) + "\n", encoding="utf-8")
        (output_dir / "logb_summary.tex").write_text("\n".join(tex_b_lines) + "\n", encoding="utf-8")


def _cmd_extract(log_file: pathlib.Path) -> int:
    try:
        mean, std = parse_logz_from_log(log_file)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"{mean}\t{std}")
    return 0


def _cmd_write(output_dir: pathlib.Path, lines: Iterable[str]) -> int:
    entries: List[Tuple[str, str, float, float, str]] = []
    for line in lines:
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            parts = line.split("\t")
            if len(parts) == 5:
                run_name, label, mean_str, std_str, identifier = parts
            elif len(parts) == 4:
                run_name, label, mean_str, std_str = parts
                identifier = ""
            else:
                continue
        except ValueError:
            continue
        entries.append((run_name, label, float(mean_str), float(std_str), identifier))
    write_tables(entries, output_dir)
    return 0


def _main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Utilities for logZ summaries.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Extract mean/std from a CLI log.")
    extract.add_argument("log_file", type=pathlib.Path)

    write = subparsers.add_parser("write", help="Write Markdown/LaTeX tables.")
    write.add_argument("--output-dir", type=pathlib.Path, required=True)

    args = parser.parse_args(argv)

    if args.command == "extract":
        return _cmd_extract(args.log_file)
    if args.command == "write":
        return _cmd_write(args.output_dir, sys.stdin)
    return parser.error("Unknown command")


if __name__ == "__main__":
    raise SystemExit(_main())
