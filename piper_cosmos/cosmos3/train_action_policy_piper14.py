"""Train Cosmos3 Piper14 action policy after registering the local experiment."""

from __future__ import annotations

import runpy

from piper_cosmos.cosmos3.local_hf_assets import bootstrap_local_hf_assets


def main() -> None:
    """Run the official Cosmos3 train module with Piper14 already registered."""

    bootstrap_local_hf_assets()
    import piper_cosmos.cosmos3.action_policy_piper14_nano  # noqa: F401

    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()
