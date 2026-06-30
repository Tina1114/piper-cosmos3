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

## 当前保留结论

- 当前正式链路是：
  - 本地 HF 资产
  - DCP 转换
  - readiness / preflight
  - SLURM SFT
- 训练和 DCP 的本地资产逻辑已经收敛到一个 helper。
- 运行时环境变量和 W&B 凭证可见性是高频故障点，必须优先检查。
- 2 卡 smoke 的价值主要是验证链路，不代表正式训练规模。
