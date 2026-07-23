# Cosmos3-Nano Piper14 Battery 20k checkpoint 训练链路

> 审计对象：`/project/peilab/wam/cosmos3_cy/reports/cosmos3_piper14/cosmos3_action/battery_piper14/battery_piper14_cosmos3_nano_20000step_4gpu_b8_acc1_offline/checkpoints/iter_000020000`  
> 仓库根目录：`/project/peilab/wam/cosmos3_cy`  
> 审计日期：2026-07-23  
> 本地 Cosmos Framework 版本：`external/cosmos` commit `60b94f685cbe8f5e0ef4209be79514d07db1566f`

## 1. 结论先行

这个 checkpoint 的训练链路是：

```text
nvidia/Cosmos3-Nano 的 HF/safetensors 目录
  external/cosmos/checkpoints/Cosmos3-Nano
        │
        │ convert_model_to_dcp（不是直接拿 HF 权重训练）
        ▼
  external/cosmos/checkpoints/Cosmos3-Nano-DCP
        │
        │ 加载 backbone；显式跳过 action head 和 net_ema
        ▼
Piper14 HDF5 → 33 帧三相机 concat_view + [当前 qpos; 32 个未来 action]
        │
        │ ActionTransformPipeline → RankPartitionedDataLoader → PackingDataLoader
        ▼
OmniMoTModel / policy 模式
  冻结 Reasoner、冻结 Wan VAE
  训练完整 Generator 路径 + 新初始化的 Piper action 投影
        │
        ▼
4-GPU FSDP，20,000 iter，最终 DCP checkpoint
```

最容易误解的三点：

1. **官方 Nano 下载后没有直接用于训练。** 本仓库先将 Hugging Face/safetensors 权重转成 Cosmos Framework 的 DCP 格式，训练实际加载的是 `Cosmos3-Nano-DCP`。
2. **Piper action head 没有从 Nano checkpoint 继承权重。** 配置在加载 Nano DCP 时显式跳过 `action2llm`、`llm2action`、`action_modality_embed`；这些参数由网络初始化函数重新初始化，再与 Generator 联合训练。
3. **“Generator 全量训练”不等于整个 Omni 模型全量训练。** 它指 Generator 路径的生成专家、时间嵌入和视频/action 输入输出投影全部进入 optimizer；Reasoner、Wan VAE 和 EMA 不训练。

## 2. checkpoint 和运行证据

运行目录保存了训练结束时的完整配置和启动信息：

- 最终 checkpoint：`reports/cosmos3_piper14/cosmos3_action/battery_piper14/battery_piper14_cosmos3_nano_20000step_4gpu_b8_acc1_offline/checkpoints/iter_000020000`
- 最终 resolved config：同级 `config.yaml`
- 启动命令和尾部 overrides：同级 `launch_info.yaml`
- Slurm 环境：同级 `job_env.yaml`
- `checkpoints/latest_checkpoint.txt` 内容为 `iter_000020000`

最终目录不是单个 `.safetensors`，而是可恢复训练的 DCP：

| 子目录 | 内容 |
|---|---|
| `model/` | `.metadata` + 4 个约 22.76 GB 的 `.distcp` shard |
| `optim/` | optimizer 元数据 + 4 个约 13.96 GB shard |
| `scheduler/` | scheduler 状态 |
| `trainer/` | trainer/iteration 状态 |

因此推理或发布前不能把这个目录当成普通 HF 模型文件；应使用 Cosmos Framework 的 DCP loader，或先通过 `cosmos_framework.scripts.export_model` 导出。

## 3. 官方 Nano 权重如何进入训练

### 3.1 本仓库实际路径

| 资产 | 本仓库实际路径 | 作用 |
|---|---|---|
| 官方 Nano HF checkpoint | `/project/peilab/wam/cosmos3_cy/external/cosmos/checkpoints/Cosmos3-Nano` | 下载后的 HF/diffusers 形态，包含 `model.safetensors.index.json` 等 |
| 转换后的 Nano DCP | `/project/peilab/wam/cosmos3_cy/external/cosmos/checkpoints/Cosmos3-Nano-DCP` | 训练实际加载的初始化权重 |
| Qwen3-VL tokenizer/processor snapshot | `/project/peilab/wam/cosmos3_cy/external/cosmos/checkpoints/hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Nano 的文本 tokenizer/Reasoner 配套资源 |
| Wan2.2 VAE | `/project/peilab/wam/cosmos3_cy/external/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth` | 将训练视频编码为 Generator latent |

HF 目录的模型卡明确标识其为 `Cosmos3-Nano`；本地 `model_index.json:L1-L27` 也显示它是 `Cosmos3OmniDiffusersPipeline`，包含 Transformer、Wan VAE、Qwen vision encoder 和 sound tokenizer 的 HF 组件声明。

### 3.2 HF → DCP，不是直接使用

转换入口：

- `scripts/convert_cosmos3_nano_to_dcp_offline.py:L24-L51`：默认输入 `external/cosmos/checkpoints/Cosmos3-Nano`，默认输出 `external/cosmos/checkpoints/Cosmos3-Nano-DCP`。
- `scripts/convert_cosmos3_nano_to_dcp_offline.py:L54-L80`：先注册本地 Qwen/Wan 资产，再调用官方 `cosmos_framework.scripts.convert_model_to_dcp.convert_model_to_dcp`。
- `scripts/run_convert_cosmos3_nano_dcp_slurm.sh:L10-L16`：固定上述三个本地路径。
- `scripts/run_convert_cosmos3_nano_dcp_slurm.sh:L38-L54`：强制离线 HF 环境，执行转换后再调用 `scripts/verify_cosmos3_dcp.py`。

本地资产替换逻辑在：

- `piper_cosmos/cosmos3/local_hf_assets.py:L17-L39`：Qwen snapshot 和 Wan VAE 的默认绝对路径。
- `piper_cosmos/cosmos3/local_hf_assets.py:L43-L57`：设置 workspace cache、`HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。
- `piper_cosmos/cosmos3/local_hf_assets.py:L95-L115`：把本地路径写入 Cosmos checkpoint registry。
- `piper_cosmos/cosmos3/local_hf_assets.py:L126-L157`：patch tokenizer 下载函数，使训练不再联网取 Qwen tokenizer。

最终 run 的 `config.yaml:L20-L32` 给出了直接证据：`load_path` 是

```text
/project/peilab/wam/cosmos3_cy/external/cosmos/checkpoints/Cosmos3-Nano-DCP
```

并且 `load_training_state: false`，说明这是**基座初始化**，不是从 Nano 的 optimizer/trainer 状态 resume。

## 4. 配置是如何组合出来的

实际配置来自三层，后面的优先级更高：

1. Python 注册实验：`piper_cosmos/cosmos3/action_policy_piper14_nano.py`
2. 启动 TOML：历史文件 `reports/cosmos3_piper14/action_policy_piper14_nano_100step_4gpu.toml`
3. 命令行尾部 overrides：20k、batch、accumulation、保存间隔等

`launch_info.yaml:L1-L14` 保存了完整组合：

```text
piper_cosmos/cosmos3/train_action_policy_piper14.py
--sft-toml=reports/cosmos3_piper14/action_policy_piper14_nano_100step_4gpu.toml
job.name=...20000step...
job.wandb_mode=offline
dataloader_train.max_samples_per_batch=8
trainer.grad_accum_iter=1
trainer.max_iter=20000
scheduler.cycle_lengths=[20000]
checkpoint.save_iter=500
```

注意：该 TOML 后来由 commit `51284dd` 从当前工作树清理，但可以从仓库提交 `776a31a` 取回：

```bash
git show 776a31a:reports/cosmos3_piper14/action_policy_piper14_nano_100step_4gpu.toml
```

它设置 4-GPU FSDP、BF16、Wan VAE、Nano DCP、100-step 初始值；尾部 override 将 run 改成最终 20,000-step。当前 `configs/cosmos3/sft/action_policy_piper14_nano.toml` 是保留下来的通用 500-step 模板，**不是该历史 run 原样使用的 TOML**。最终事实应以 run 内 `config.yaml` 为准。

## 5. 从 Slurm 到 Cosmos Framework 的调用链

| 顺序 | 文件及代码 | 责任 |
|---:|---|---|
| 1 | `scripts/submit_cosmos3_piper14_sft.sh:L31-L44` | 运行 readiness，检查数据、DCP、VAE 和 Slurm account |
| 2 | `scripts/submit_cosmos3_piper14_sft.sh:L46-L55` | `sbatch` 提交 4-GPU 作业并导出路径 |
| 3 | `scripts/run_cosmos3_piper14_sft_slurm.sh:L10-L18` | 解析 TOML、数据、Qwen、本地输出和 GPU 数 |
| 4 | `scripts/run_cosmos3_piper14_sft_slurm.sh:L36-L47` | 强制要求 DCP、Wan VAE、Qwen snapshot 存在 |
| 5 | `scripts/run_cosmos3_piper14_sft_slurm.sh:L70-L95` | 设置离线环境，用 `torch.distributed.run` 启动 Piper 训练 module |
| 6 | `piper_cosmos/cosmos3/train_action_policy_piper14.py:L10-L16` | 注册本地 HF 资产和 Piper 实验，然后 `runpy` 调官方 `cosmos_framework.scripts.train` |
| 7 | `external/cosmos/packages/cosmos3/cosmos_framework/scripts/train.py:L227-L298` | 读取 TOML，叠加尾部 override，构造最终 Hydra config |
| 8 | `external/cosmos/packages/cosmos3/cosmos_framework/scripts/train.py:L179-L224` | 初始化分布式、实例化 `OmniMoTModel` 和 dataloader，调用 `trainer.train` |

这里没有另写一套训练循环；本仓库做的是**注册 Piper 数据/域/config，然后把控制权交给官方 Cosmos Framework trainer**。

## 6. Piper14 action 是怎样接入的

### 6.1 没有独立的“Piper action-head 类”

Piper 接入由两部分组成：

1. 本仓库注册 Piper embodiment：
   - `piper_cosmos/cosmos3/domain.py:L6-L8`：`piper14`、domain ID `21`、raw action dim `14`。
   - `piper_cosmos/cosmos3/domain.py:L11-L17`：写入官方 `EMBODIMENT_TO_DOMAIN_ID` 和 `EMBODIMENT_TO_RAW_ACTION_DIM`。
2. 官方通用 action head 根据 domain ID 选择对应权重：
   - `external/cosmos/packages/cosmos3/cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py:L188-L214`：创建 `action2llm`、`llm2action` 和 `action_modality_embed`。
   - `external/cosmos/packages/cosmos3/cosmos_framework/model/vfm/mot/domain_aware_linear.py:L17-L47`：每个 embodiment domain 有自己的 weight/bias embedding。
   - `external/cosmos/packages/cosmos3/cosmos_framework/model/vfm/mot/domain_aware_linear.py:L49-L78`：forward 时按每个 token 的 `domain_id` 取对应矩阵。

也就是说，“Piper action head”实际是：

```text
Piper 注册文件 domain.py
  + Cosmos3 通用 DomainAwareLinear action2llm/llm2action
  + checkpoint 中 domain_id=21 对应的参数切片
```

### 6.2 14D action 的定义

`configs/dataset_configs/battery_assemble_hdf5.yaml:L29-L50` 定义 action：

- 类型：`absolute_joint_position_command`
- 维度：14
- 顺序：左臂 7 维在前，右臂 7 维在后；每臂为 6 个 joint + gripper
- 对齐：`action_t_close_to_qpos_t_plus_1`

训练没有做 action normalization：

- 最终 `config.yaml:L56-L68`：`action_normalization: null`
- `piper_cosmos/cosmos3/piper14_hdf5_action_dataset.py:L162-L163`：若请求 normalization 会直接报 `NotImplementedError`

Transform 会把 14D 补零到 `max_action_dim=64`，并记录 `raw_action_dim=14` 供 loss mask 使用；相关逻辑入口是 `ActionTransformPipeline` 的 `action_processor`，见
`external/cosmos/packages/cosmos3/cosmos_framework/data/vfm/action/transforms.py:L426-L433` 和 `L663-L669`。所以补出的 50 个通道不是监督目标。

### 6.3 action head 如何初始化

最终 config 的 `keys_to_skip_loading` 位于 `config.yaml:L20-L25`：

```yaml
- net_ema.
- action2llm
- llm2action
- action_modality_embed
- action_pos_embed
```

因此这次训练不是“用 Nano checkpoint 初始化 Piper head”，而是：

1. 先按 Nano 架构建立 action head；
2. 加载 Nano DCP 时跳过全部 action head；
3. `cosmos3_vfm_network.py:L241-L257` 用 truncated normal/zero 初始化 `action2llm`、`llm2action`、`action_modality_embed`；
4. 训练时给 action 模块 5 倍 nominal learning rate。

最终 20k checkpoint 的 `model/.metadata` 证明这些参数已被保存：

| 参数 | shape |
|---|---:|
| `net.action_modality_embed` | `[4096]` |
| `net.action2llm.fc.weight` | `[32, 262144]` |
| `net.action2llm.bias.weight` | `[32, 4096]` |
| `net.llm2action.fc.weight` | `[32, 262144]` |
| `net.llm2action.bias.weight` | `[32, 64]` |

其中 `32` 是 embodiment domain 数，`4096` 是 Nano hidden size，`262144 = 64 × 4096`。配置中虽保留 `action_pos_embed` 的 skip pattern，但该 run 使用 `unified_3d_mrope`，网络在 `cosmos3_vfm_network.py:L208-L210` 将独立 `action_pos_embed` 设为 `None`，最终 metadata 中也没有该参数。

## 7. 数据和 dataloader 链路

### 7.1 原始 HDF5

实际训练根目录由最终 config 固化为：

```text
/project/peilab/wam/physical_WM/data/battery_assemble/perfect
```

HDF5 key 在 `configs/dataset_configs/battery_assemble_hdf5.yaml:L11-L20`：

- 三路 RGB：`cam_high`、`cam_left_wrist`、`cam_right_wrist`
- action：`/action`
- current state：`/observations/qpos`

### 7.2 每个样本如何构造

`piper_cosmos/cosmos3/piper14_hdf5_action_dataset.py` 是本仓库接入真实 Piper 数据的核心文件：

- `L29-L66`：只接受 `concat_view`、`policy`、`use_state=True`。
- `L82-L101`：取 `qpos[t]`、`action[t:t+32]`、33 帧视频，并拼成 `[qpos; future_actions]`，所以 action tensor 是 `[33,14]`。
- `L103-L118`：按 episode 建滑窗索引，stride 为 1。
- `L120-L131`：每路取 33 帧；两路 wrist 缩为一半，左右横拼后放在 high camera 下方。

原始单路图像是 `480×640`，concat 后是：

```text
cam_high 480×640
cam_left 240×320 | cam_right 240×320
--------------------------------------
原始 concat canvas = 720×640
```

随后 `ActionTransformPipeline` 以 `resolution="480"` 查最接近的支持比例，实际选择 `640×640` bucket，并做等比例 resize + reflection padding；算法见
`external/cosmos/packages/cosmos3/cosmos_framework/data/vfm/action/transforms.py:L43-L81`、`L329-L380`。因此不要把 `720×640` 原始 canvas 与模型实际 tokenization 尺寸混为一谈。

`configs/dataset_configs/battery_assemble_hdf5.yaml:L57-L63` 里仍写有 `action_horizon: 16`，但这个字段没有被该 dataset factory 用作 chunk 长度；最终 resolved config 的 `chunk_length: 32`（`config.yaml:L56-L67`）才是这次 run 的实际值。这是当前配置中一个值得清理的陈旧字段。

### 7.3 Cosmos 数据包装和分布式加载

完整链路：

```text
Piper14HDF5ActionDataset
  → ActionTransformPipeline
  → ActionSFTDataset
  → ActionIterableShuffleDataset
  → RankPartitionedDataLoader
  → PackingDataLoader
  → OmniMoTModel.training_step
```

对应证据：

- `piper14_hdf5_action_dataset.py:L141-L192`：创建 raw dataset、官方 transform、`ActionSFTDataset` 和 iterable episode shuffle。
- `external/cosmos/packages/cosmos3/cosmos_framework/data/vfm/action/datasets/action_sft_dataset.py:L25-L42`：每次 `__getitem__` 应用 transform。
- 同文件 `L46-L87`：按 rank × worker 对 episode block 做互斥分片，episode 顺序打乱、episode 内顺序读取。
- `piper_cosmos/cosmos3/action_policy_piper14_nano.py:L113-L152`：`PackingDataLoader` 包 `RankPartitionedDataLoader`；每 rank 最多 pack 8 个样本，4 workers。
- 最终 `config.yaml:L47-L90`：该 run 的真实 dataloader resolved values。

Transform 顺序在
`external/cosmos/packages/cosmos3/cosmos_framework/data/vfm/action/transforms.py:L558-L586`、`L601-L669`：

1. resize/pad 视频；
2. 给 caption 追加 viewpoint、时长/FPS、分辨率信息；
3. Qwen tokenizer 编码文本，并以 `cfg_dropout_rate=0.1` 做 text CFG dropout；
4. 根据 `mode` 建 `SequencePlan`；
5. action 补到 64D并记录真实维度。

这次历史 run 使用普通文本 prompt + metadata append，不是后来官方 Edge/DROID recipe 使用的 JSON prompt。

## 8. policy 模式的输入、监督和输出

`build_sequence_plan_from_mode` 的定义在
`external/cosmos/packages/cosmos3/cosmos_framework/data/vfm/action/transforms.py:L235-L326`。

对本 run：

- `video_length=33`
- `action_length=33`，因为第 0 行是当前 qpos，后面 32 行是未来 action
- `mode="policy"`

因此 SequencePlan 为：

- text：条件；
- vision：第 0 帧 clean，后 32 帧加噪并预测；
- action：第 0 行 current qpos clean，后 32 行加噪并预测。

可以概括成：

```text
输入条件：
  指令文本
  当前三相机 concat_view
  当前 14D qpos

训练输出/监督：
  32-step、14D 的未来 absolute joint action
  同时间范围的辅助未来视频 latent
```

网络用 `action2llm` 把 noisy action token 投到 MoT hidden space，叠加 action modality/timestep/3D position 信息；见
`cosmos3_vfm_network.py:L740-L799`。再用 `llm2action` 把 noisy action 位置的 hidden state 解码回 64D，见 `L801-L858`，loss 只对真实 14D channel 生效。

训练是 joint video-action flow matching，不是分类或自回归 action token prediction：

- `OmniMoTModel._compute_losses` 在 `external/cosmos/packages/cosmos3/cosmos_framework/model/vfm/omni_mot_model.py:L1048-L1069` 计算 vision flow-matching loss。
- 同文件 `L1073-L1092` 计算 action flow-matching loss，并乘 `action_loss_weight=10`。

推理时可以只取预测 action，不解码辅助未来视频。本仓库部署 wrapper 在
`piper_cosmos/deployment/cosmos_piper14_policy.py:L152-L180` 将当前 state 写入第 0 行，执行联合采样后只返回第 1–32 行的 14D action。

## 9. Reasoner 在哪里冻结，Generator 在哪里训练

### 9.1 冻结 Reasoner 的直接证据

冻结不是由 `model.config.vlm_config.model_instance.config.freeze_und` 完成的。最终 config 里它反而是 `false`（`config.yaml:L243-L252`）。真正的冻结发生在 optimizer 参数白名单：

`piper_cosmos/cosmos3/action_policy_piper14_nano.py:L39-L55`：

```python
keys_to_select=[
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "action2llm",
    "llm2action",
    "action_modality_embed",
]
```

实现位于
`external/cosmos/packages/cosmos3/cosmos_framework/utils/vfm/optimizer.py:L76-L127`，其文档明确说明非匹配参数全部冻结；实际赋值在 `L129-L140`：

```python
if ... not any(key in pn for key in keys_to_select):
    p.requires_grad = False
```

Reasoner 参数名不匹配上述 Generator/action 白名单，所以被冻结。它仍参与 forward，提供文本条件和 Reasoner→Generator 的上下文，不等于从计算图中删除。

### 9.2 “Generator 全量训练”的准确含义

最终 `config.yaml:L333-L356` 保存了同一 allowlist。进入 optimizer 的生成路径包括：

- `moe_gen`：每层 Generator 专属 attention/MLP/norm 投影；
- `time_embedder`：扩散时间条件；
- `vae2llm`、`llm2vae`：视频 latent 与 hidden space 的双向投影；
- 三个 action 模块。

因此这里是**完整 Generator 路径 SFT + action head SFT**，不是 LoRA（`config.yaml:L168-L171` 显示 `lora_enabled: false`），也不是全模型 Reasoner+Generator SFT。

Wan VAE 本身被冻结：`external/cosmos/packages/cosmos3/cosmos_framework/model/vfm/tokenizers/wan2pt2_vae_4x16x16.py:L1155-L1175` 将其设为 `eval().requires_grad_(False)`，encode 也带 `@torch.no_grad()`。EMA 网络在 `external/cosmos/packages/cosmos3/cosmos_framework/model/vfm/omni_mot_model.py:L358-L371` 中 `requires_grad_(False)`，通过主网络更新，不进入 optimizer。

### 9.3 nominal LR 与调度后的 LR

- Generator 参数 nominal LR：`2e-4`
- action head nominal LR：`2e-4 × 5 = 1e-3`
- scheduler：`LambdaLinear`，`f_start=0`、`f_max=0.4`、`f_min=0`、cycle 20,000、warmup 0

所以 scheduler 的最大倍率是 0.4；对应最大实际 LR 分别约为 `8e-5` 和 `4e-4`，随后线性降至 0。不能只看 optimizer 的 `lr=2e-4` 就把它当作整个 run 的实际峰值。

## 10. 对应 Cosmos3 technical report 的框架

官方 [Cosmos3 Technical Report](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) 的模型架构部分（PDF 第 7–11 页）和 Generator/policy 后训练部分（第 29–32 页）将 Cosmos3 描述为双塔 Mixture-of-Transformers：

- Reasoner：autoregressive tower，处理离散文本/理解上下文；
- Generator：diffusion tower，对连续的 vision/action 等模态做联合去噪；
- Generator 能关注 Reasoner 和 Generator token，Reasoner 保持 causal；
- policy 后训练冻结 Reasoner，只更新 Generator 专属参数及新 action 投影；
- policy 同时预测未来 action 和辅助未来 RGB。

其中第 9 页明确说明 domain-specific action input/output projection 从头初始化并与 MoT 联合优化；第 32 页给出的 Nano-DROID policy 又明确采用 fresh action encoder/decoder/embedding、action 参数 5× LR、32-step absolute joint action、4-step inference、shift 5 和 CFG 3。这些论文结论与本 run 的 skip list、optimizer allowlist 和推理默认值逐项一致。

本 run 的对应关系如下。

| Technical report 模块 | 本 run 是否使用 | 本地证据与说明 |
|---|---|---|
| Omni/MoT 主模型 | 使用 | `config.yaml:L118-L122` 为 `cosmos_framework.model.vfm.omni_mot_model.OmniMoTModel` |
| AR Reasoner tower | forward 使用、参数冻结 | Qwen3-VL-8B 文本 backbone；optimizer allowlist 不包含 Reasoner 参数 |
| Diffusion Generator tower | 使用并训练 | `moe_gen` 全部进入 optimizer |
| Wan2.2 4×16×16 VAE | 使用但冻结 | 将 33 帧视频变成 latent；`requires_grad_(False)` |
| vision input/output projection | 使用并训练 | `vae2llm`、`llm2vae` |
| domain-aware action encoder/decoder | 使用并训练 | `action2llm`、`llm2action`，Piper domain ID 21 |
| action modality embedding | 使用并训练 | `action_modality_embed` |
| rectified-flow vision/action objective | 使用 | vision loss scale 10；action loss weight 10 |
| unified 3D mRoPE / timestep embedding | 使用 | `config.yaml:L137-L151`；`time_embedder` 训练 |
| two-way joint attention | 使用 | `config.yaml:L161` |
| Qwen Reasoner ViT visual encoder | **本训练路径不使用** | `external/cosmos/packages/cosmos3/cosmos_framework/model/vfm/mot/unified_mot.py:L254-L263` 的 `include_visual` 默认 false；数据 transform 只提供 `text_token_ids` 和 Generator `video`，视觉观测走 Wan VAE，不走 Reasoner ViT |
| sound/audio tokenizer 和 head | **不使用** | `sound_gen: false`、`sound_tokenizer: null` |
| Reasoner next-token/CE 训练 | **不使用** | 这是 VFM policy flow-matching SFT，不计算语言 CE |
| forward dynamics / inverse dynamics | **不使用** | dataset `L44-L49` 只允许 `policy` |
| image2video、T2V、V2V 等模式 | **不使用** | 单一 Piper policy 数据源 |
| LoRA | **不使用** | `lora_enabled: false` |

需要特别区分“ViT 不使用”和“VAE 使用”：当前视觉条件确实参与训练，但走的是 Generator 的 Wan VAE latent 路径；配置里出现 `Qwen/Qwen3-VL-8B-Instruct` 主要用于 Nano Reasoner/文本 processor，不能据此断言 Qwen ViT 在这次 policy SFT 中被调用。

## 11. 最终后训练参数

以下均来自该 run 保存的 `config.yaml`，不是当前模板的默认值。

| 类别 | 实际值 |
|---|---|
| 模型 | Cosmos3-Nano `OmniMoTModel` |
| 初始化 | `external/cosmos/checkpoints/Cosmos3-Nano-DCP` |
| checkpoint load | 跳过 EMA 和全部 action head；不加载 training state |
| 训练模式 | `policy` |
| 数据 | Battery Assemble `perfect` HDF5 |
| action | 14D absolute joint command，raw/un-normalized |
| state | 14D current qpos，第 0 action row |
| horizon | 32 future actions；视频共 33 帧 |
| FPS | 30 |
| view | `concat_view`，三相机 |
| 原始/模型 canvas | 原始 `720×640`；480 tier 实际 `640×640` |
| max action dim | 64，真实 14D mask |
| CFG text dropout | 0.1 |
| loss | rectified flow；vision `loss_scale=10`；action `action_loss_weight=10` |
| time sampling | action/image `logitnormal`，video `waver`，uniform weighting |
| optimizer | FusedAdam，betas `(0.9,0.99)`，eps `1e-8`，weight decay `0.05` |
| nominal LR | Generator `2e-4`；action head `1e-3` |
| scheduler | LambdaLinear，cycle 20,000，factor `0.4 → 0`，无 warmup |
| GPU/并行 | 4-GPU FSDP，shard degree 4，replicate degree 1 |
| precision | BF16；FSDP master dtype FP32 |
| activation checkpoint | full，保留 `fmha` ops |
| per-rank packed samples | 最大 8 |
| grad accumulation | 1 |
| 有效 count-based global batch 上限 | `8 × 4 × 1 = 32` samples/update |
| iteration | 20,000 |
| grad clip | 1.0，force finite |
| logging/save | log 每 10 iter；DCP 每 500 iter |
| validation | 关闭 |
| seed | 42 |
| W&B | offline |
| EMA | 开启，rate 0.1；EMA 参数不反传 |

## 12. 复现时应保留的边界

1. 用 run 内 `config.yaml` 复核最终事实，不要只用当前 500-step TOML。
2. base init 必须指向转换后的 DCP，不能把 HF `Cosmos3-Nano` 目录直接填给该训练入口。
3. `domain_id=21`、14D 顺序、raw absolute action、qpos 第 0 行必须在训练和推理端一致。
4. `chunk_length=32` 是真实值；数据 YAML 中 `action_horizon:16` 是未被该 loader 消费的陈旧配置。
5. 推理预处理必须复用相同 `concat_view` 和 480-tier resize/pad；不能只喂 high camera。
6. 该 run 没有 validation；`iter_000020000` 证明训练状态完整，不等价于证明它是真机成功率最好的 checkpoint，仍应通过离线 action 误差、时序平滑、安全边界和真机分阶段评估选择部署版本。

## 13. 外部权威资料

- [Cosmos3 Technical Report](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)
- [NVIDIA Cosmos3-Nano model card](https://huggingface.co/nvidia/Cosmos3-Nano)
- [官方 Cosmos3-Nano action-policy fine-tuning cookbook](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/generator/action/finetune/README.md)
- [官方 Cosmos Framework DROID post-training guide](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/action_policy_droid_posttrain.md)
