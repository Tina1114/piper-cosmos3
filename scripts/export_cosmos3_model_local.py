#!/usr/bin/env python3
"""Export a Cosmos3 DCP checkpoint to a Hugging Face model directory.

This variant accepts a local Qwen snapshot path for the visual tower weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import safetensors.torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

from cosmos_framework.checkpoint.dcp import CustomLoadPlanner
from cosmos_framework.inference.common.args import CheckpointOverrides
from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.inference.common.config import serialize_config_dict
from cosmos_framework.inference.common.init import init_script, is_rank0
from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel
from cosmos_framework.model.vfm.omni_mot_model import OmniMoTModel
from cosmos_framework.scripts.export_model import (
    _coerce_to_base_model,
    _load_safetensor_weights,
    _rewrite_visual_fqns_for_vfm,
)
from cosmos_framework.utils import log
from cosmos_framework.utils.lazy_config.registry import convert_target_to_string
from cosmos_framework.inference.common.public_model_config import build_public_model_config
from piper_cosmos.cosmos3.local_hf_assets import bootstrap_local_hf_assets


init_script(
    env={
        "COSMOS_DEVICE": "cpu",
        "COSMOS_TRAINING": "1",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--vit-path", type=Path, required=True, help="Local Qwen snapshot directory.")
    parser.add_argument("--config-only", action="store_true")
    return parser.parse_args()


def export_model_local(args: argparse.Namespace) -> None:
    register_checkpoints()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_overrides = CheckpointOverrides(
        checkpoint_path=str(args.checkpoint_path),
        config_file=str(args.config_file),
    )
    checkpoint_args = checkpoint_overrides.build_checkpoint(checkpoints={})

    log.info("Loading config...")
    model_dict = checkpoint_args.load_model_config_dict()
    if not model_dict["config"]["ema"]["enabled"]:
        checkpoint_args.use_ema_weights = False
    model_dict["config"]["ema"]["enabled"] = False

    vit_path = args.vit_path.resolve()
    if not vit_path.is_dir():
        raise ValueError(f"Local VLM checkpoint directory does not exist: {vit_path}")
    bootstrap_local_hf_assets(qwen_snapshot=vit_path)

    log.info("Loading model...")
    _coerce_to_base_model(model_dict)
    model_dict["_target_"] = convert_target_to_string(OmniMoTModel)
    hf_config = Cosmos3OmniConfig(model=build_public_model_config(model_dict))
    hf_config.save_pretrained(args.output_dir)
    hf_model = Cosmos3OmniModel(hf_config)

    log.info("Saving model...")
    if not args.config_only:
        storage_reader = FileSystemReader(str(args.checkpoint_path))
        state_dict = get_model_state_dict(hf_model.model)
        dcp.load(
            state_dict=state_dict,
            storage_reader=storage_reader,
            planner=CustomLoadPlanner(load_ema_to_reg=checkpoint_args.use_ema_weights),
        )
        state_dict = get_model_state_dict(
            hf_model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )
        if not is_rank0():
            return

        vit_state_dict = _load_safetensor_weights(vit_path, lambda x: x.startswith("model.visual."))
        if not vit_state_dict:
            raise ValueError(f"No vision weights found in local VLM checkpoint: {vit_path}")
        state_dict.update(_rewrite_visual_fqns_for_vfm(vit_state_dict))

        hf_model.save_pretrained(
            args.output_dir,
            state_dict=state_dict,
        )

    hf_config_file = args.output_dir / "config.json"
    hf_config_json = json.loads(hf_config_file.read_text())
    hf_config_json["model_type"] = "cosmos3_omni"
    serialize_config_dict(hf_config_json, hf_config_file)

    serialize_config_dict(checkpoint_args.model_dump(mode="json"), args.output_dir / "checkpoint.json")
    print(f"Saved model to {args.output_dir}")


def main() -> None:
    export_model_local(parse_args())


if __name__ == "__main__":
    main()
