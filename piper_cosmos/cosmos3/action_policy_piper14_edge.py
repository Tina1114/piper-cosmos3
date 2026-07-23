"""Cosmos3-Edge action-policy SFT experiment for Battery Piper14."""

from __future__ import annotations

import copy
from pathlib import Path

import cosmos_framework
from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L
from piper_cosmos.cosmos3.edge_training_audit import ACTION_HEAD_KEYS, EdgeTrainingAuditCallback
from piper_cosmos.cosmos3.piper14_hdf5_action_dataset import get_piper14_hdf5_sft_dataset

cs = ConfigStore.instance()

action_policy_piper14_edge = copy.deepcopy(action_policy_droid_nano)
action_policy_piper14_edge["job"] = dict(
    project="cosmos_battery",
    group="edge_piper14",
    name="battery_piper14_cosmos3_edge_base_fresh_head_20k",
    wandb_mode="online",
)
action_policy_piper14_edge["model"]["config"] = copy.deepcopy(EDGE_MODEL_CONFIG)
action_policy_piper14_edge["optimizer"]["keys_to_select"] = [
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "k_norm_und_for_gen",
    "action2llm",
    "llm2action",
    "action_modality_embed",
]
action_policy_piper14_edge["scheduler"]["cycle_lengths"] = [20_000]
action_policy_piper14_edge["trainer"]["grad_accum_iter"] = 1
action_policy_piper14_edge["trainer"]["logging_iter"] = 10
action_policy_piper14_edge["trainer"]["max_iter"] = 20_000
action_policy_piper14_edge["checkpoint"]["save_iter"] = 500
action_policy_piper14_edge["trainer"]["callbacks"]["edge_runtime_audit"] = L(EdgeTrainingAuditCallback)(
    report_path="${oc.env:EDGE_TRAIN_AUDIT_REPORT,/tmp/cosmos3_edge_training_audit.json}",
    selected_keys="${optimizer.keys_to_select}",
    action_head_keys=list(ACTION_HEAD_KEYS),
)
action_policy_piper14_edge["dataloader_train"] = L(PackingDataLoader)(
    audio_sample_rate=48_000,
    dataset_name="action_piper14_edge",
    max_samples_per_batch=8,
    max_sequence_length=None,
    patch_spatial=2,
    sound_latent_fps=0,
    tokenizer_spatial_compression_factor=16,
    tokenizer_temporal_compression_factor=4,
    dataloader=L(RankPartitionedDataLoader)(
        batch_size=1,
        in_order=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=4,
        sampler=None,
        datasets=dict(
            piper14=dict(
                ratio=1,
                dataset=L(get_piper14_hdf5_sft_dataset)(
                    root="${oc.env:PIPER14_ROOT}",
                    config_path="${oc.env:PIPER14_DATA_CONFIG}",
                    fps=30.0,
                    chunk_length=32,
                    mode="policy",
                    use_state=True,
                    iterable_shuffle=True,
                    episode_shuffle_seed=42,
                    action_normalization=None,
                    viewpoint="concat_view",
                    resolution="480",
                    max_action_dim="${model.config.max_action_dim}",
                    cfg_dropout_rate=0.1,
                    tokenizer_config="${model.config.vlm_config.tokenizer}",
                    format_prompt_as_json=True,
                ),
            ),
        ),
    ),
)

edge_model_config = action_policy_piper14_edge["model"]["config"]
framework_package_root = Path(cosmos_framework.__file__).resolve().parent
edge_model_config["vlm_config"]["model_instance"]["config"]["base_config"]["json_file"] = str(
    framework_package_root
    / "model/generator/reasoner/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json"
)
edge_model_config["action_gen"] = True
edge_model_config["vision_gen"] = True
edge_model_config["sound_gen"] = False
edge_model_config["resolution"] = "480"
edge_model_config["tokenizer"]["encode_exact_durations"] = [33]
edge_model_config["max_num_tokens_after_packing"] = -1
edge_model_config["rectified_flow_training_config"]["loss_scale"] = 10.0
edge_model_config["rectified_flow_training_config"]["action_loss_weight"] = 10.0

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_piper14_edge",
    node=action_policy_piper14_edge,
)
