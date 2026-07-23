# Cosmos3 Piper14 训练排坑经验简报

日期：2026-06-30

这份简报只保留后续仍有复用价值的经验，不记录一次性作业号和中间态日志。

## 1. 训练和 DCP 必须共用同一套本地 HF 资产逻辑

- 问题：DCP 转换已经本地化，但训练入口仍可能在线访问 `Qwen/Qwen3-VL-8B-Instruct` 或 `Wan2.2_VAE.pth`。
- 结论：训练和 DCP 转换必须共用 `piper_cosmos/cosmos3/local_hf_assets.py`。
- 原因：如果两条链路各维护一套本地化逻辑，后续很容易漂移，导致“DCP 能过、训练不能过”。

## 2. 运行时要明确走本地优先和离线保护

- 问题：节点代理异常时，训练初始化会因为 tokenizer 或 VAE 解析回到线上而失败。
- 结论：运行脚本应统一导出：
  - `HF_HOME`
  - `IMAGINAIRE_CACHE_DIR`
  - `UV_CACHE_DIR`
  - `UV_TOOL_DIR`
  - `XDG_DATA_HOME`
  - `XDG_CACHE_HOME`
  - `COSMOS3_QWEN_SNAPSHOT`
  - `COSMOS3_WAN_VAE_PATH`
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`
- 原则：运行时不要依赖代理和外网，优先消费本地资产。

## 3. tokenizer patch 只能 patch 下载入口，不能替换构造路径

- 问题 1：手工构造 tokenizer 会丢失 `chat_template` 等配置。
- 问题 2：把 `_target_` 换成本地闭包函数会导致对象不可 pickle。
- 结论：只 patch `download_tokenizer_files` 这类下载入口，仍让官方 `from_pretrained(local_path)` 完成 tokenizer 构造。

## 4. 不要在 batch 脚本里覆盖用户原始 `HOME`

- 问题：脚本若默认 `HOME=${REPO_ROOT}`，batch 环境会看不到用户原本 `~/.netrc`。
- 后果：`wandb.init(...)` 可能报没有 API key，即使登录节点本来已有凭证。
- 结论：优先保留原始 `HOME`，仅在未设置时再回退；需要时再额外支持 `WANDB_API_KEY_FILE`。

## 5. readiness / preflight 要把 W&B 可见性前置检查掉

- 问题：如果 `wandb_mode=online`，很多错误会在训练初始化较晚阶段才暴露。
- 结论：`scripts/preflight_cosmos3_piper14_sft.py` 和 `scripts/cosmos3_piper14_readiness.py` 应前置检查 batch 可见凭证来源。
- 至少要支持检查以下之一：
  - `WANDB_API_KEY`
  - `WANDB_API_KEY_FILE`
  - `HOME` 下的 `.netrc`

## 6. 2 卡 smoke 和正式可训练不是同一个问题

- 现象：2 卡可能已经通过 dataset、forward、backward，但在 `optimizer.step()` OOM。
- 结论：这说明训练链路基本打通，但显存配置还不够。
- 做法：不要把 2 卡 OOM 误判为“整个训练方案不可用”；应继续用 4 卡或更高并行度验证。

## 7. Slurm 资源申请要和训练并行度同步

- 问题：只改 `NPROC_PER_NODE` 而不改 `sbatch --gres` / `--cpus-per-task`，会导致资源声明和实际训练配置脱节。
- 结论：提交脚本里的 `SLURM_GPUS`、`SLURM_CPUS` 应与 `NPROC_PER_NODE` 联动。

## 8. CPU affinity 问题本质上是 cpuset 约束问题

- 现象：多卡初始化时可能在 `distributed.init()` 阶段触发 `sched_setaffinity` 错误。
- 根因：请求 affinity 时没有和当前 Slurm 作业允许的 cpuset 取交集。
- 结论：affinity 逻辑必须限制在当前作业可用 cpuset 内；完全不相交时要回退到 allowed cpuset。

### 8.1 Edge framework 仍需要移植这项修复

2026-07-24 已重新拉取并检查 NVIDIA 官方
`cosmos-framework` 的 `origin/main`：

- 官方最新检查点：
  `f734253f0f6af3e268372402f44435c38f55ef3e`
  （`Export Cosmos3 Edge scheduler metadata (#130)`，2026-07-23）。
- 最新
  `cosmos_framework/utils/device.py::Device.get_cpu_affinity()`
  仍然只返回 NVML 给出的 CPU 列表，没有与进程当前允许的 cpuset
  求交集。
- 最新
  `cosmos_framework/utils/distributed.py::init()`
  仍然直接执行
  `os.sched_setaffinity(0, device.get_cpu_affinity())`，并且只捕获
  `pynvml.NVMLError`，没有处理 cpuset 不匹配产生的 `OSError`。
- 官方新增的
  `cosmos_framework/utils/distributed_test.py`
  只验证 CPU checkpoint conversion 选择 Gloo backend，没有覆盖
  Slurm/cgroup cpuset 不相交的情况。

因此，这不是已经被 Edge 官方 framework 替代的旧补丁。对于本项目使用
Slurm/cgroup 资源隔离的多 GPU 后训练，仍建议保留；否则 Edge 训练仍可能在
进入 NCCL 初始化之前就因为 CPU affinity 非法而退出。

### 8.2 从 Nano framework 移植到 Edge framework 的最小范围

Nano 修复已经保存在：

- 外层可追踪补丁：
  `patches/cosmos-framework/0001-fix-respect-Slurm-cpuset-for-CPU-affinity.patch`
- Nano framework 本地分支：
  `local/slurm-cpuset-affinity`
- Nano framework 修复提交：
  `b4795a9`
- 修复基线：
  `90cd348877c37b888942c988b631eb1611bf2950`

Edge framework 不应直接强行应用旧 patch。新版
`cosmos_framework/utils/distributed.py` 已经加入 CPU checkpoint conversion
所需的 Gloo backend 逻辑，旧 patch 的上下文与新版代码不再完全一致。正确的
小型移植只有两处：

1. 在 Edge 的
   `cosmos_framework/utils/device.py`
   中移植 `resolve_cpu_affinity()`：
   去重 NVML 请求列表，与 `os.sched_getaffinity(0)` 返回的 allowed cpuset
   求交集；请求为空或完全不相交时回退到 allowed cpuset。
2. 在 Edge 的
   `cosmos_framework/utils/distributed.py::init()`
   中只替换“Set GPU affinity”代码块：
   先读取 requested/allowed affinity，再调用
   `resolve_cpu_affinity()`，仅在结果非空时调用
   `os.sched_setaffinity()`，并同时捕获
   `pynvml.NVMLError` 和 `OSError`。

以下 Edge 官方逻辑必须原样保留，不能被 Nano patch 覆盖：

- `COSMOS_DEVICE=cuda` 使用 NCCL；
- `COSMOS_DEVICE=cpu` 使用 Gloo；
- CPU checkpoint conversion 依赖 Gloo 同步 CPU 上的 tokenizer/model
  tensor；
- 新版 process-group timeout 和初始化日志。

### 8.3 Edge 移植后的验证门槛

移植完成后至少做三层验证：

1. 复用
   `tests/test_device_cpu_affinity.py`
   的四种情况：请求完全合法、部分相交、完全不相交、allowed cpuset
   不可获取。
2. 为 Edge 的
   `cosmos_framework/utils/distributed_test.py`
   增加一项测试，确认 `sched_setaffinity()` 收到的是过滤后的 CPU
   集合，同时保留现有 Gloo backend 测试。
3. 在正式 Edge 后训练前运行一次 2/4 GPU Slurm smoke test，确认日志已经
   越过 affinity、process-group 初始化、dataset 首批读取和第一次
   forward/backward。

只有单元测试通过还不足以证明集群拓扑正确；最终应以真实 Slurm
作业中的 `os.sched_getaffinity(0)`、过滤后的 target affinity 和多进程初始化
结果为准。

## 当前保留结论

- 当前正式链路是：
  - 本地 HF 资产
  - DCP 转换
  - readiness / preflight
  - SLURM SFT
- 训练和 DCP 的本地资产逻辑已经收敛到一个 helper。
- 运行时环境变量和 W&B 凭证可见性是高频故障点，必须优先检查。
- 2 卡 smoke 的价值主要是验证链路，不代表正式训练规模。
