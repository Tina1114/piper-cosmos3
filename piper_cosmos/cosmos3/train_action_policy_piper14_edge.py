"""Train Cosmos3-Edge Piper14 after installing strict local asset routing."""

from __future__ import annotations

import runpy

from piper_cosmos.cosmos3.local_edge_assets import bootstrap_local_edge_assets


def main() -> None:
    bootstrap_local_edge_assets()
    import piper_cosmos.cosmos3.action_policy_piper14_edge  # noqa: F401

    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()
