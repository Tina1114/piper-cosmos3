# Cosmos3 Piper14 Inference Deployment

这个分支只保留 Cosmos3 Piper14 battery_assemble 推理部署所需文件。

## 保留范围

- 启动 Cosmos3 Piper14 policy server。
- 通过 RPC 请求 `[32, 14]` action chunk。
- 用 HDF5 真实帧运行无真机 dry-run RTC。
- 为后续真机接入保留 `ObservationSource` / `ActionSink` 边界。

训练、DCP 转换、checkpoint export、offline eval、Forward Dynamics、历史报告和测试文件不属于这个分支。

## 核心入口

启动 policy server：

```bash
PYTHONPATH=/project/peilab/wam/cosmos3_cy:/project/peilab/wam/cosmos3_cy/external/cosmos/packages/cosmos3 \
external/cosmos/packages/cosmos3/.venv/bin/python \
scripts/serve_cosmos_piper14_policy.py \
  --checkpoint /project/peilab/wam/cosmos3_cy/cosmos_battery/18k \
  --host 0.0.0.0 \
  --port 8766
```

运行无真机 dry-run RTC：

```bash
PYTHONPATH=/project/peilab/wam/cosmos3_cy \
python scripts/dry_run_piper14_rtc_runtime.py \
  --host <gpu-node-hostname> \
  --port 8766 \
  --data-root /project/peilab/wam/physical_WM/data/battery_assemble/perfect \
  --data-config configs/dataset_configs/battery_assemble_hdf5.yaml \
  --steps 64 \
  --control-hz 30 \
  --replan-interval 8
```

更多说明见：

```text
docs/cosmos3_piper14_policy_deployment.md
docs/inference_assets_manifest.md
docs/instruction_cache_optimization_zh.md
docs/cosmos3_piper14_inference_profiling.md
```
