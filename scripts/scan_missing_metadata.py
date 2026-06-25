#!/usr/bin/env python3
"""Scan likely project files for missing Piper metadata evidence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


TERMS = {
    "fps_or_timing": ["fps", "rate", "frequency", "sleep", "dt"],
    "gripper": ["gripper", "gripper_val", "gripper_val_mutiple", "joint_states"],
    "task_instruction": ["Put the three objects", "instruction", "task"],
    "success_split": ["success", "perfect"],
    "cameras": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
}

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".launch",
    ".xml",
    ".sh",
    ".cfg",
    ".ini",
}


@dataclass
class Match:
    path: Path
    line_no: int
    line: str


def is_text_candidate(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TEXT_SUFFIXES


def iter_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            candidates = [root]
        else:
            candidates = [p for p in root.rglob("*") if is_text_candidate(p)]
        for path in candidates:
            resolved = path.resolve()
            if resolved not in seen and is_text_candidate(path):
                seen.add(resolved)
                files.append(path)
    return sorted(files)


def scan_file(path: Path, terms: list[str], max_matches: int) -> list[Match]:
    matches: list[Match] = []
    lowered_terms = [term.lower() for term in terms]
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f, start=1):
                lowered = line.lower()
                if any(term.lower() in lowered for term in lowered_terms):
                    matches.append(Match(path=path, line_no=idx, line=line.strip()))
                    if len(matches) >= max_matches:
                        break
    except OSError:
        return []
    return matches


def scan(roots: list[Path], max_matches_per_file: int) -> dict[str, list[Match]]:
    files = iter_files(roots)
    results: dict[str, list[Match]] = {category: [] for category in TERMS}
    for path in files:
        for category, terms in TERMS.items():
            results[category].extend(scan_file(path, terms, max_matches_per_file))
    return results


def format_report(results: dict[str, list[Match]], roots: list[Path]) -> str:
    lines = [
        "# Missing Metadata Scan",
        "",
        "## Scanned Roots",
        "",
    ]
    for root in roots:
        status = "exists" if root.exists() else "missing"
        lines.append(f"- `{root}`: {status}")

    for category, matches in results.items():
        lines.extend(["", f"## {category}", ""])
        if not matches:
            lines.append("UNKNOWN")
            continue
        for match in matches:
            lines.append(f"- `{match.path}:{match.line_no}`: {match.line}")

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search repository and data-adjacent files for FPS, gripper, task, success, and camera metadata."
    )
    parser.add_argument("--repo-root", type=Path, required=True, help="Repository or source tree root to scan.")
    parser.add_argument("--data-root", type=Path, required=True, help="Data root to scan for metadata files.")
    parser.add_argument("--output", type=Path, required=True, help="Markdown output path.")
    parser.add_argument(
        "--max-matches-per-file",
        type=int,
        default=20,
        help="Limit matches per category per file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [args.repo_root, args.data_root]
    results = scan(roots, args.max_matches_per_file)
    report = format_report(results, roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
