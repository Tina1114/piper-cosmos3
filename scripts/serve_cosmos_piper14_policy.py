#!/usr/bin/env python3
"""Start the RTC-style Cosmos Piper14 policy server."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from piper_cosmos.deployment.cosmos_piper14_policy import CosmosPiper14PolicyConfig
from piper_cosmos.deployment.cosmos_piper14_policy_server import serve_cosmos_piper14_policy
from piper_cosmos.cosmos3.local_hf_assets import WAN_VAE_REPOSITORY, bootstrap_local_hf_assets


def materialize_runtime_config(config_file: str | None, *, wan_vae_path: Path) -> str | None:
    """Create a location-independent config with an absolute, validated VAE path."""
    if config_file is None:
        return None
    source = Path(config_file).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    try:
        tokenizer_config = payload["model"]["config"]["tokenizer"]
        vlm_tokenizer_config = payload["model"]["config"]["vlm_config"]["tokenizer"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unsupported Cosmos config structure in {source}") from exc

    tokenizer_config["vae_path"] = str(wan_vae_path.resolve())
    # Keep the portable repository ID here. bootstrap_local_hf_assets patches
    # Cosmos' tokenizer resolver to map it to the validated local snapshot.
    vlm_tokenizer_config["pretrained_model_name"] = "Qwen/Qwen3-VL-8B-Instruct"

    runtime_dir = Path(os.environ.get("TMPDIR", "/tmp")).expanduser()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    destination = runtime_dir / f"cosmos_piper14_runtime_config_{os.getpid()}.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cosmos-piper14-policy-server] Runtime config: {destination}", flush=True)
    print(f"[cosmos-piper14-policy-server] Wan VAE: {wan_vae_path}", flush=True)
    return str(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT / "cosmos_battery" / "20k"))
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--authkey", default="cosmos-piper14")
    parser.add_argument("--prompt", default="Assemble the mouse's battery.")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--max-action-dim", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--resolution", default="480")
    parser.add_argument(
        "--condition-only-vae",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Encode only the clean first video frame and synthesize generated-frame latent placeholders.",
    )
    parser.add_argument(
        "--instruction-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse fixed-prompt Reasoner K/V from the second policy chunk onward.",
    )
    parser.add_argument(
        "--instruction-cache-dir",
        default=None,
        help="Optional persistent cache directory. Omit it for safer process-local memory caching only.",
    )
    parser.add_argument("--instruction-cache-max-entries", type=int, default=4)
    parser.add_argument("--mock-backend", action="store_true")
    parser.add_argument("--timing", action="store_true", help="Print synchronized per-stage inference timings.")
    parser.add_argument("--cuda-memory", action="store_true", help="Print per-stage CUDA allocator usage.")
    parser.add_argument(
        "--cuda-memory-history",
        default=None,
        help="Dump one inference CUDA allocator history to this .pickle file.",
    )
    parser.add_argument("--cuda-memory-history-max-entries", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_paths = bootstrap_local_hf_assets()
    runtime_config_file = materialize_runtime_config(
        args.config_file,
        wan_vae_path=repository_paths[WAN_VAE_REPOSITORY],
    )
    config = CosmosPiper14PolicyConfig(
        checkpoint=args.checkpoint,
        config_file=runtime_config_file,
        prompt=args.prompt,
        action_horizon=args.action_horizon,
        max_action_dim=args.max_action_dim,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        resolution=args.resolution,
        num_steps=args.num_steps,
        guidance=args.guidance,
        shift=args.shift,
        fps=args.fps,
        seed=args.seed,
        condition_only_vae=args.condition_only_vae,
        instruction_cache=args.instruction_cache,
        instruction_cache_dir=args.instruction_cache_dir,
        instruction_cache_max_entries=args.instruction_cache_max_entries,
        host=args.host,
        port=args.port,
        mock_backend=args.mock_backend,
        timing=args.timing,
        cuda_memory=args.cuda_memory,
        cuda_memory_history=args.cuda_memory_history,
        cuda_memory_history_max_entries=args.cuda_memory_history_max_entries,
    )
    serve_cosmos_piper14_policy(config, host=args.host, port=args.port, authkey=args.authkey)


if __name__ == "__main__":
    main()
