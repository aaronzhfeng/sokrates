#!/usr/bin/env python3
"""
Inline \\input{...} directives to produce a single-file AAAI-compliant .tex.

AAAI AuthorKit 2026 states the camera-ready source (excluding .bib) must be a
single .tex file and that \\input should not be used in the final submission.

Usage:
  python3 paper/tools/inline_tex.py \
    --input paper/sokrates.tex \
    --output paper/sokrates_single.tex
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


INPUT_RE = re.compile(r"^(\s*)\\input\{([^}]+)\}\s*$")


def _resolve_input_path(current_file: Path, spec: str) -> Path:
    # Resolve relative to the file that contains the \input.
    p = (current_file.parent / spec).resolve()
    if p.suffix == "":
        p = p.with_suffix(".tex")
    return p


def inline_file(path: Path, seen: set[Path]) -> str:
    path = path.resolve()
    if path in seen:
        raise RuntimeError(f"Detected recursive input cycle at: {path}")
    seen.add(path)

    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        m = INPUT_RE.match(raw_line.rstrip("\n"))
        if not m:
            lines.append(raw_line)
            continue

        indent, spec = m.group(1), m.group(2)
        child = _resolve_input_path(path, spec)
        if not child.exists():
            raise FileNotFoundError(f"\\input{{{spec}}} resolved to missing file: {child}")

        rel = child.relative_to(Path.cwd()) if child.is_absolute() else child
        lines.append(f"{indent}% === BEGIN INPUT: {rel} ===\n")
        lines.append(inline_file(child, seen))
        if not lines[-1].endswith("\n"):
            lines.append("\n")
        lines.append(f"{indent}% === END INPUT: {rel} ===\n")

    seen.remove(path)
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to root .tex (with \\input).")
    parser.add_argument("--output", type=str, required=True, help="Path to write single-file .tex.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    content = inline_file(input_path, seen=set())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


