# Cosmos3 Piper14 推理耗时与显存分析

本文记录 Piper14 policy server 的分段计时、CUDA 显存采样方法，以及 2026-07-21 在 RTX 4090 上的基线结果。目标是在 HDF5 dry-run 与真机 RPC 路径中使用同一套口径，避免把 CUDA 异步等待误判为数据预处理。

## 1. 测量范围

一次 `infer` 按以下层次记录：

```text
机器人/HDF5 观测
  -> RPC send
  -> observation 校验
  -> 三视角拼接、PNG/base64
  -> 33 帧 tensor、ActionTransformPipeline、batch
  -> Cosmos prepare：condition-only VAE（1 帧）或完整 VAE（33 帧）、文本、sequence packing、noise/mask
  -> Reasoner UND prefill
  -> UniPC 4-step denoise
  -> action 截取、GPU->CPU、校验
  -> RPC send response
```

服务器计时在 GPU 阶段前后执行 `torch.cuda.synchronize()`。没有同步时，CUDA kernel 可能在 `.cpu().numpy()` 才真正等待，导致采样耗时被错误归入后处理。

日志中的包含关系：

- `model.generate.total` 包含 `model.prepare.total`、prefill、denoise 和模型内部后处理。
- `model.prepare.total` 包含 vision/action、text 和 pack。
- `model.denoise.velocity[N]` 中的 `N` 是 forward 次数。`guidance=3` 时，4 个 denoise step 通常对应 conditional/unconditional 共 8 次 forward。
- 包含项不能直接相加。

## 2. 启动服务

4090 24GB 使用当前 framework 的 reasoner/VAE offload。以下命令显式开启分段测时、逐阶段 CUDA 显存和第一条请求的 allocator history：

```bash
COSMOS3_REASONER_OFFLOAD=1 \
COSMOS3_VAE_CPU_OFFLOAD=1 \
bash scripts/start_cosmos_piper14_20k_server.sh \
  --no-instruction-cache \
  --timing \
  --cuda-memory \
  --cuda-memory-history /tmp/cosmos_piper14_cuda_memory_snapshot.pickle
```

先关闭 instruction cache，避免固定 prompt K/V 复用干扰数据处理优化的测量。分别启动以下两组服务：

```text
完整 VAE 基线：--no-condition-only-vae --no-instruction-cache
首帧 VAE 优化：--condition-only-vae --no-instruction-cache
```

两组必须使用相同 observation、seed、steps、guidance、shift 和 offload 环境变量。每组至少记录第一条 cold 请求和 3 条 warm 请求。instruction cache 应在 VAE 对比完成后单独测试，只比较第 2 条及后续 warm 请求。

即使启用 condition-only VAE，Piper wrapper 目前仍会创建并 transform 33 帧 RGB tensor；优化发生在 framework 的 VAE encode 阶段。日志应出现 `condition-only VAE fast path enabled`，否则该请求已经回退到完整 VAE。

history 在模型加载完成后开启，只记录第一条 inference；snapshot 写完后自动关闭。`_dump_snapshot()` 会增加数秒落盘/序列化时间，所以生产延迟应看第二条及后续 warm 请求。

## 3. 客户端与真机

HDF5 dry-run：

```bash
PYTHONPATH=<piper-repo> <python> scripts/dry_run_piper14_rtc_runtime.py \
  --host <server-host> \
  --port 8766 \
  --data-root <hdf5-root> \
  --data-config configs/dataset_configs/battery_assemble_hdf5.yaml \
  --episode <episode.hdf5> \
  --steps 65 \
  --chunk-size 32 \
  --replan-interval 32 \
  --timing
```

真机 runtime 若未暴露 `--timing` 参数，可通过环境变量启用 RPC 计时，无需修改真机控制逻辑：

```bash
COSMOS_PIPER14_CLIENT_TIMING=1 <real-runtime-command>
```

真机和服务器日志需按同一个请求顺序对齐：

```text
[cosmos-piper14-client-timing]  send / recv_wait / total
[cosmos-piper14-timing]         服务器各阶段 / policy.total
[cosmos-piper14-rpc-timing]     server dispatch / response send
```

三张 `480x640x3 uint8` 原图约 2.64MiB。localhost 不能代表真机网络；真机侧重点查看客户端 `send`。`recv_wait` 包含服务器完整推理，不是纯网络耗时。

## 4. CUDA 显存口径

`--cuda-memory` 输出：

- `baseline_allocated`：模型加载后、请求开始时 PyTorch 活跃显存。
- `baseline_reserved`：PyTorch caching allocator 已保留显存。
- `baseline_driver_used`：驱动层总占用，包含非 PyTorch/CUDA context。
- `request_peak`：本次请求的 PyTorch allocated 高水位。
- `delta_from_baseline`：阶段结束时相对常驻基线的增量。
- `new_peak`：该阶段把本次请求高水位推高了多少。

`torch.cuda.memory._record_memory_history()` 生成 allocator 事件、allocation/free 时间线和调用栈。它只覆盖 PyTorch CUDA allocator，不能解释 NCCL、CUDA context 或其他库的全部显存，因此需要和 `nvidia-smi`、`driver_used` 一起看。

可视化 snapshot：

```bash
<cosmos3-python> -m torch.cuda._memory_viz trace_plot \
  -o /tmp/cosmos_piper14_memory_trace.html \
  /tmp/cosmos_piper14_cuda_memory_snapshot.pickle

<cosmos3-python> -m torch.cuda._memory_viz segment_plot \
  -o /tmp/cosmos_piper14_memory_segments.html \
  /tmp/cosmos_piper14_cuda_memory_snapshot.pickle
```

Snapshot 是 pickle 文件，只应打开自己生成或可信来源的文件。

## 5. 4090 完整 VAE 基线结果

以下结果测于 condition-only VAE 合入前，可作为 `--no-condition-only-vae --no-instruction-cache` 对照。测试输入来自 `episode_5.hdf5`，三路图像均为 `480x640 uint8`，state 为 14 维 float32；checkpoint 为 20k，action chunk 为 `[32,14]`。

### Warm 请求

| 阶段 | 耗时 | policy.total 占比 |
|---|---:|---:|
| 普通数据预处理 | 194.07ms | 3.27% |
| Cosmos prepare | 580.81ms | 9.78% |
| Reasoner prefill，2 次 | 1902.46ms | 32.03% |
| Denoise velocity，8 次 | 3232.46ms | 54.43% |
| Action 后处理与返回 | 小于 1ms | 小于 0.02% |
| 服务器 `policy.total` | 5939.01ms | 100% |
| 客户端端到端 | 5942.41ms | - |

普通数据预处理细分：

| 阶段 | 耗时 |
|---|---:|
| concat view | 3.11ms |
| PNG/base64 | 76.66ms |
| repeat 33 帧 | 1.56ms |
| ActionTransformPipeline | 112.59ms |
| batch build | 0.014ms |
| vision/action prepare，主要为 VAE | 569.65ms |

### 冷请求

第一条请求中，Cosmos prepare 为 7.57s、reasoner prefill 为 27.29s、denoise 为 3.29s。开启 memory history 时，snapshot dump 额外消耗约 7s；因此记录到的 46.68s 端到端不能作为生产冷启动延迟。比较优化效果时至少记录第一条 cold 和一条 warm，稳定性测试建议记录 3 至 10 条 warm。

### 显存

| 指标 | 显存 | 24GB 可用显存占比 |
|---|---:|---:|
| 模型常驻 allocated | 15703.5MiB | 65.23% |
| PyTorch reserved | 18094.0MiB | 75.16% |
| 驱动层 used | 19086.2MiB | 79.28% |
| 请求峰值 allocated | 17712.7MiB | 73.57% |
| 单次请求新增峰值 | 2009.2MiB | 8.35% |

Allocator history 对约 1996MiB 瞬时增量的归因：

- VAE 临时张量约 1841MiB，占 92.3%。
- video normalization 约 155MiB，占 7.7%。
- 最大单次 VAE allocation 约 265MiB。
- reasoner 逐层 H2D 可见约 96MiB allocation，但流式释放，没有超过 VAE 高水位。

## 6. 与真机 14.6s 记录的关系

真机截图中的“输入编码、数据准备、后处理、RPC 约 11-12s”是 `14.6s - Sampling 约 3s` 得到的剩余值，不是各阶段直接测量结果。旧路径只在 RTC runtime 外层统计 `policy.infer()`，服务器内部没有分段计时，也没有 CUDA 同步。

真机输入经过 `cv_bridge` 后同样被转换为连续 `uint8` ndarray；ROS 图像同步和机器人状态读取发生在 `policy.infer()` 计时开始前。因此下一次真机测试应以新增日志为准，优先排查：

1. 客户端 RPC `send` 是否因网络产生秒级延迟。
2. 请求是 server cold 还是 warm。
3. `model.prepare.vision_action` 是否命中 condition-only VAE，以及相对完整 VAE 的耗时和峰值变化。
4. `model.reasoner.prefill` 是否与当前 1.9s warm 基线一致。
5. 旧采样计时是否因 CUDA 异步把等待归入后处理。

只有真机服务器输出逐阶段日志后，才能判断额外的 8 至 9 秒究竟来自网络、VAE、reasoner，还是旧计时口径。

## 7. 推送范围

本次 profiling 功能只需要以下文件：

```text
.gitignore
piper_cosmos/deployment/cosmos_piper14_policy.py
piper_cosmos/deployment/cosmos_piper14_policy_server.py
piper_cosmos/deployment/cosmos_piper14_remote_client.py
scripts/serve_cosmos_piper14_policy.py
scripts/dry_run_piper14_rtc_runtime.py
tests/test_cosmos_piper14_backend_imports.py
tests/test_cosmos_piper14_policy_preprocess.py
docs/cosmos3_piper14_inference_profiling.md
```

不要提交 checkpoint、HDF5、snapshot pickle、memory HTML、运行日志和本地绝对路径修正版 config。
