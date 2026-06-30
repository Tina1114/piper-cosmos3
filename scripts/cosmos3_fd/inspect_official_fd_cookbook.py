#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path


KEYWORDS = (
    "forward_dynamics",
    "droid_lerobot",
    "action_chunk_size",
    "vision_path",
    "action_path",
    "domain_name",
    "model_mode",
)

SEARCH_FILES = (
    "README.md",
    "cookbooks/cosmos3/README.md",
    "cookbooks/cosmos3/generator/action/README.md",
    "cookbooks/cosmos3/generator/action/run_fd_with_cosmos_framework.ipynb",
    "cookbooks/cosmos3/generator/action/run_fd_with_vllm.ipynb",
)

COOKBOOK = Path("cookbooks/cosmos3/generator/action/run_fd_with_cosmos_framework.ipynb")


@dataclass(frozen=True)
class KeywordHit:
    keyword: str
    path: Path
    line_number: int
    line: str


def read_git_head(cosmos_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(cosmos_root), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except OSError:
        pass

    head_path = cosmos_root / ".git" / "HEAD"
    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = head.removeprefix("ref: ").strip()
        return (cosmos_root / ".git" / ref).read_text(encoding="utf-8").strip()
    return head


def find_hits(cosmos_root: Path) -> list[KeywordHit]:
    hits: list[KeywordHit] = []
    seen_keywords: set[str] = set()
    for relative in SEARCH_FILES:
        path = cosmos_root / relative
        if not path.is_file():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for keyword in KEYWORDS:
                if keyword in seen_keywords:
                    continue
                if keyword in line:
                    hits.append(
                        KeywordHit(
                            keyword=keyword,
                            path=relative,
                            line_number=line_number,
                            line=line.strip()[:300],
                        )
                    )
                    seen_keywords.add(keyword)
    return hits


def inspect_official_checkout(repo_root: Path) -> tuple[str, Path, str]:
    cosmos_root = repo_root / "external" / "cosmos"
    if not cosmos_root.is_dir():
        raise FileNotFoundError(f"missing official Cosmos checkout: {cosmos_root}")

    cookbook_path = cosmos_root / COOKBOOK
    if not cookbook_path.is_file():
        raise FileNotFoundError(f"missing official FD cookbook: {cookbook_path}")

    commit = read_git_head(cosmos_root)
    hits = find_hits(cosmos_root)
    found = {hit.keyword for hit in hits}
    missing = [keyword for keyword in KEYWORDS if keyword not in found]
    if missing:
        raise RuntimeError(f"missing keyword evidence in official files: {', '.join(missing)}")

    lines = [
        "# Official Cosmos3 DROID Forward Dynamics References",
        "",
        f"official_cosmos_root: {cosmos_root}",
        f"official_cosmos_commit: {commit}",
        f"official_fd_cookbook: {cookbook_path}",
        "",
        "## Keyword Evidence",
        "",
    ]
    for hit in hits:
        lines.append(f"- `{hit.keyword}`: `{hit.path}:{hit.line_number}`")
        lines.append(f"  {hit.line}")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This inspection is CPU-only and does not load checkpoints.",
            "- Heavy FD inference must be submitted through SLURM on a compute node.",
        ]
    )
    return commit, cookbook_path, "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the official NVIDIA Cosmos3 DROID forward-dynamics cookbook."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Current project repository root. Defaults to this script's grandparent.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/cosmos3_fd/official_fd_references.txt"),
        help="Evidence report path. Relative paths are resolved under --repo-root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else repo_root / args.output
    commit, cookbook_path, report = inspect_official_checkout(repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"official_cosmos_commit: {commit}")
    print(f"official_fd_cookbook: {cookbook_path}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
