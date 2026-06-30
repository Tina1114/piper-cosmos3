#!/usr/bin/env python3
"""Episode split helpers shared by training, eval, and export scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_episode_split(path: Path | str | None) -> dict[str, set[str]]:
    if path is None:
        return {}
    split_path = Path(path)
    if not split_path.exists():
        return {}
    with split_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    raw_splits = payload.get("splits", {}) if isinstance(payload, dict) else {}
    if not isinstance(raw_splits, dict):
        return {}
    return {
        str(name): {str(Path(item)) for item in paths}
        for name, paths in raw_splits.items()
        if isinstance(paths, list)
    }


def dataset_indices_for_split(dataset: Any, split: dict[str, set[str]], split_name: str) -> list[int]:
    allowed = split.get(split_name, set())
    if not allowed:
        return []
    selected: list[int] = []
    for idx, sample_index in enumerate(getattr(dataset, "index", [])):
        episode_path = str(Path(getattr(sample_index, "episode_path")))
        if episode_path in allowed:
            selected.append(idx)
    return selected
