# Cosmos3-Edge 4B × Piper14 Battery 后训练迁移方案

> 目标：用官方新发布的 `nvidia/Cosmos3-Edge`，在同一 Battery Assemble Piper14 真机 HDF5 数据上训练 14D 双臂 policy。  
> 方案日期：2026-07-23。Edge 是 2026-07-20 发布的新模型，官方代码仍在快速变化，因此本文固定审计快照。  
> 官方 `NVIDIA/cosmos` 快照：`7c22d8aa2e97adcd7857399d1ff1b088be1ff401`  
> 官方 `NVIDIA/cosmos-framework` 快照：`f734253f0f6af3e268372402f44435c38f55ef3e`

## 1. 决策摘要

### 1.1 现有 Piper action head 能否复用

要分“代码”和“权重”回答：

- **Piper14 接口和 head 机制可以复用。**  
  `piper_cosmos/cosmos3/domain.py` 中的 14D/domain 注册，以及 Cosmos Framework 的通用 `DomainAwareLinear(action2llm/llm2action)` 设计都适用于 Edge。
- **Nano 20k checkpoint 中已训练的 Piper action-head 权重不能直接加载到 Edge。**  
  Nano hidden size 是 4096，Edge hidden size 是 2048；action 投影矩阵的 shape 绑定 hidden size，`state_dict` 不能对齐。
- **推荐 Edge 第一版从官方 Edge base 初始化 backbone，Piper action head 重新随机初始化。**  
  保留 Nano 成功 recipe 的“跳过 action head + 5× action nominal LR”策略。不要把单臂 DROID policy checkpoint 当作双臂 Piper 的默认初始化。

### 1.2 是否已有官方 Edge action-policy 后训练 recipe

截至本文快照：

- 官方已发布 [Cosmos3-Edge base](https://huggingface.co/nvidia/Cosmos3-Edge)；
- 官方已发布 [Cosmos3-Edge-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID) 和推理 server；
- Cosmos Framework 已有 `EDGE_MODEL_CONFIG` 和 Edge **vision SFT**；
- 但公开的 [action-policy fine-tuning cookbook](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/generator/action/finetune/README.md) 仍只列 Nano 的 DROID、LIBERO-10、LIBERO-all；
- 官方注册的 robot action experiment 仍是 `action_policy_droid_nano`，内部显式导入 `NANO_MODEL_CONFIG`。

所以本项目不能声称“直接使用官方 Edge action SFT recipe”。正确做法是：

```text
官方 Edge 模型定义/视觉 SFT
  + 官方 Nano action-policy recipe
  + 本仓库已验证的 Piper14 数据/action 合约
  = 新的 Piper14 Edge action-policy experiment
```

## 2. Edge 与当前 Nano run 的关键差异

| 项目 | 当前 Piper Nano | Edge 4B | 迁移影响 |
|---|---:|---:|---|
| 总模型规模 | 16B | 4B | Edge 面向低延迟/边缘部署 |
| Reasoner/Generator hidden size | 4096 | 2048 | Nano action-head 权重 shape 不兼容 |
| Reasoner backbone | Qwen3-VL-8B | Nemotron-2B-Dense-VL | 不能复用 Qwen-specific bootstrap |
| vision encoder | Qwen ViT（本 policy 训练未启用） | SigLIP2/Nemotron VL 配套 | policy 视觉仍主要走 Wan VAE Generator 路径 |
| native Generator resolution | Nano config 默认 720；本数据用 480 tier | 480 | Piper 的 480-tier 数据设置适配 Edge |
| sound | Nano base 支持，本 run 关闭 | Edge 不支持 sound | 保持 `sound_gen=false` |
| action | base 支持 | base 支持 | 仍需注册 Piper 14D domain |
| video-to-video transfer | 支持 | 官方说明暂不支持 | 与 policy 无关，不应加入任务 |

官方 Edge model card 将其描述为 4B MoT，由 AR Reasoner 和 diffusion Generator 两个互补 tower 组成，并支持 text/image/action 输入与 action/video 输出；只测试 BF16。详见
[Cosmos3-Edge model card](https://huggingface.co/nvidia/Cosmos3-Edge)。

Edge hidden size 的代码证据是官方
`cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json:L1-L30`，其中 `hidden_size=2048`。当前 Nano 20k checkpoint metadata 中：

```text
action_modality_embed       [4096]
action2llm.fc.weight        [32, 64×4096]
llm2action.fc.weight        [32, 4096×64]
```

Edge 对应 shape 应是：

```text
action_modality_embed       [2048]
action2llm.fc.weight        [32, 64×2048]
llm2action.fc.weight        [32, 2048×64]
```

这就是 Nano Piper action 权重不能直接复用的结构性原因，不是 checkpoint 格式问题。

## 3. “Piper action head”具体是哪几个文件

当前仓库没有一个叫 `PiperActionHead` 的独立类。它由以下文件共同实现：

| 文件 | 作用 | Edge 迁移 |
|---|---|---|
| `piper_cosmos/cosmos3/domain.py:L6-L17` | 注册 `piper14 → domain_id 21 → raw dim 14` | 逻辑直接复用；import namespace 要适配新版 |
| `piper_cosmos/cosmos3/piper14_hdf5_action_dataset.py:L82-L101` | 构造 `[当前 qpos; 32 future action]` | 直接复用数据语义 |
| `external/cosmos/packages/cosmos3/cosmos_framework/model/vfm/mot/domain_aware_linear.py:L17-L78` | 每个 domain 一套 action 投影 | 新版路径变为 `cosmos_framework/model/generator/mot/domain_aware_linear.py`，机制相同 |
| `external/cosmos/packages/cosmos3/cosmos_framework/model/vfm/mot/cosmos3_vfm_network.py:L188-L214` | 建 `action2llm`、`llm2action`、`action_modality_embed` | Edge 仍由通用网络按 hidden size 新建 |
| `piper_cosmos/cosmos3/action_policy_piper14_nano.py:L39-L55` | 把 action 模块加入 optimizer，设 5× LR | 作为 Edge experiment 的直接参考 |

新版官方 `domain_utils.py` 当前已占用的 domain ID 包括 0–9、12、13、15、16、20，**21 当前没有冲突**。但 domain ID 是 checkpoint 语义的一部分；启动前仍应在固定的新版 framework commit 上做一次 collision assertion，不能假定官方以后永远不使用 21。

## 4. Edge base action 权重该怎样处理

官方新版 `EDGE_MODEL_CONFIG` 的注释说明，2026-07-16 更新后的 Edge checkpoint 已包含 action-head 权重；旧版没有这些权重并可能产生 NaN。对应文件：

`cosmos_framework/configs/base/experiment/sft/models/edge_model_config.py:L34-L39`

这不意味着它已经有 Piper14 head：

- 官方 model card 列出的 robot embodiment 以 DROID/UR/Fractal/Bridge/UMI、dual Franka、AgiBot 等为主，没有 Piper14；
- Piper 的 domain 21 是本仓库新增语义；
- 14D 双 Piper absolute joint command 与 DROID 单臂 8/10D action 不是同一个控制空间；
- 即使 tensor 有 domain 21 这一行，也没有公开证据证明那一行代表 Piper。

因此建议：

### 默认实验：fresh Piper head

加载 Edge DCP 时跳过：

```python
[
    "net_ema.",
    "action2llm",
    "llm2action",
    "action_modality_embed",
    "action_pos_embed",
]
```

优点是语义干净，与官方 Nano DROID recipe 和本仓库 Nano 20k run 一致。

### 可选第二组 ablation：保留通用 action embedding，仅重置 domain 21

如果后续要验证 Edge base 的 action pretraining 是否有迁移价值，可以：

1. 加载 Edge 的 `action_modality_embed` 和整个 action head；
2. 只随机重置 `action2llm/llm2action` 的 domain 21 weight/bias 行；
3. 确认其他 domain 永远不会出现在 Piper batch；
4. 与 fresh-head baseline 做同数据、同 seed 对照。

这需要新增 selective row reset 代码；当前 checkpoint skip 机制是按参数名跳过整块 tensor，不能直接表达“只跳过第 21 行”。它不应成为第一版必须项。

## 5. 初始化 checkpoint 的选择

### 推荐：`nvidia/Cosmos3-Edge` base

理由：

- 用户目标是复刻 Piper14 的 base→policy 后训练链路；
- Edge base 官方定位就是 downstream action task 的初始化；
- action head 会按 Piper 语义新建；
- 可以与 Nano base→Piper20k 做较干净的 backbone 对照。

本地建议路径：

```text
/project/peilab/wam/cosmos3_cy/external/cosmos/checkpoints/Cosmos3-Edge
/project/peilab/wam/cosmos3_cy/external/cosmos/checkpoints/Cosmos3-Edge-DCP
```

仍需先 HF/safetensors → DCP，不能把 HF 目录直接填给当前 SFT checkpointer。官方通用转换入口是：

```bash
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path /project/peilab/wam/cosmos3_cy/external/cosmos/checkpoints/Cosmos3-Edge \
  -o /project/peilab/wam/cosmos3_cy/external/cosmos/checkpoints/Cosmos3-Edge-DCP
```

转换必须由**支持 Edge 的新版 Cosmos Framework**执行。

### 不推荐作为默认：`nvidia/Cosmos3-Edge-Policy-DROID`

它可以作为额外实验，但不应成为默认：

- DROID 是单 Franka、单臂 action contract；
- 官方 Edge-DROID 的 domain/action head 语义不等于 Piper 双臂 14D；
- 相机排列、FPS、prompt schema、action horizon 均可能不同；
- 直接继承 DROID policy 会同时改变 backbone 初始化和 task-domain 初始化，难以判断收益来源。

如果要做，仍应重置 Piper domain 投影，并把结果标为“DROID-policy transfer ablation”，不能与 Edge base baseline 混写。

## 6. 哪些现有内容可以直接复用

### 6.1 数据语义可直接复用

- `configs/dataset_configs/battery_assemble_hdf5.yaml`
  - HDF5 keys；
  - 左 7D + 右 7D action 顺序；
  - absolute joint command；
  - qpos/state 语义。
- `piper_cosmos/cosmos3/piper14_hdf5_action_dataset.py`
  - episode/window 索引；
  - 33 帧采样；
  - high + two wrist 的 concat；
  - `[qpos; 32 actions]`；
  - episode-block streaming shuffle。
- 数据根：
  `/project/peilab/wam/physical_WM/data/battery_assemble/perfect`
- `mode="policy"`、`use_state=true`、`fps=30`、`chunk_length=32`、`raw_action_dim=14`、`max_action_dim=64`。

注意仍要清理 YAML 中未被 loader 消费的 `training.action_horizon:16`，或至少在新 Edge config 中明确 runtime 的 32 才是唯一真值。

### 6.2 训练策略可直接复用

- policy-only，而不是随机混合 forward/inverse/policy；
- Reasoner 冻结，完整 Generator 路径训练；
- action head fresh init；
- action loss weight 10；
- action head nominal LR 为 Generator 的 5×；
- BF16、FSDP、full activation checkpointing；
- DCP base init，`load_training_state=false`；
- 每 500 iter 保存；
- 4-step policy inference、跳过辅助视频 decode。

### 6.3 部署接口可复用

`piper_cosmos/deployment/cosmos_piper14_policy.py` 中以下逻辑与 backbone 无关：

- `CosmosPiper14PolicyConfig`；
- 三相机 `compose_concat_view`；
- 14D state/action 检查；
- 第 0 action row 写当前 qpos；
- 只返回未来 32×14 action；
- RTC client/server 接口。

但该文件内的旧 `data.vfm` import、LIBERO service adapter 和 plain prompt transform 需要适配新版 Edge framework，不能原样认为已兼容。

## 7. 哪些文件必须新增或改写

当前环境实际有三层 Git 边界：

```text
cosmos3_cy                                      # 本项目，master
└── external/cosmos                            # NVIDIA/cosmos @ 60b94f...
    └── packages/cosmos3                       # NVIDIA/cosmos-framework @ 90cd348...
```

真正提供 `cosmos_framework.*` 训练代码的是第三层 `external/cosmos/packages/cosmos3`，不是第二层 cookbook 仓库。当前外层仓库把 `external/cosmos` 记录为 gitlink，但没有 `.gitmodules`；第二层 `external/cosmos/.gitignore` 又忽略了整个 `packages/`，所以最外层 Git **不会自动记录 framework 的 commit 或本地修改**。Nano 的 CPU-affinity 修复现已提交到 framework 本地分支，并另存到外层可追踪 patch：

```text
framework branch: local/slurm-cpuset-affinity
framework commit: b4795a9
outer patch: patches/cosmos-framework/0001-fix-respect-Slurm-cpuset-for-CPU-affinity.patch
```

这意味着只在最外层建立 Edge branch，不能自动隔离或锁定 framework；只在 framework 内建立 Edge branch，又不能管理本项目的 Edge config、dataset adapter 和启动脚本。两层都需要明确管理。

当前 cookbook commit `60b94f685cbe8f5e0ef4209be79514d07db1566f` 和 framework commit `90cd348877c37b888942c988b631eb1611bf2950` 都早于 Edge 发布，并没有完整的新版 Edge 训练支持。因此只把 TOML 中 `load_path` 改成 Edge 是不够的。

### 推荐的可维护布局

不要在同一个物理目录里反复切换 Nano/Edge 的三层 branch。推荐使用外层 Git worktree：

```text
/project/peilab/wam/cosmos3_cy
  ├── 外层 branch: master
  ├── external/cosmos: 固定 Nano cookbook commit
  └── external/cosmos/packages/cosmos3: 固定 Nano framework commit + 已审计 patch

/project/peilab/wam/cosmos3_cy_edge
  ├── 外层 branch: feature/cosmos3-edge-piper14
  ├── external/cosmos: 固定支持 Edge 的 cookbook commit
  └── external/cosmos/packages/cosmos3: 固定支持 Edge 的 framework commit
```

外层 Edge branch 从 `master` 创建，但 Git branch 只复制提交历史指针，不会复制 checkpoint 大文件。两个 worktree 共享最外层 Git object database，却拥有各自工作目录，因此 Nano 和 Edge 的 framework、venv、配置不会在切 branch 时互相覆盖。

每个外层 branch 还应提交一个依赖锁定文件，例如：

```text
configs/dependencies/cosmos3_nano.lock.yaml
configs/dependencies/cosmos3_edge.lock.yaml
```

至少记录：

- `NVIDIA/cosmos` URL + commit；
- `NVIDIA/cosmos-framework` URL + commit；
- Python/CUDA/PyTorch 版本；
- 本地 framework patch 的文件和 hash；
- checkpoint、Wan VAE 和 processor 的版本/路径。

本次核查已确认 Edge 仍需要 cpuset 小型移植。因此第三层应从固定的官方
Edge commit 建本地修复分支，并把最终 Edge patch 保存在本项目的
`patches/cosmos-framework/`；不要只留下无法追踪的第三层 dirty working
tree。

### 7.1 必须先升级并固定 framework

建议在独立分支/独立 external checkout 上固定支持 Edge 的 commit，不要直接覆盖 Nano 20k 的可复现环境。至少需要新版：

- `cosmos_framework/configs/base/experiment/sft/models/edge_model_config.py`
- `cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/configs/Nemotron-2B-Dense-VL.json`
- `cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/nemotron_3_dense_vl.py`
- `cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/vision_siglip2.py`
- 新版 Edge processor/checkpoint conversion
- 新 `data.generator` / `model.generator` namespace

Edge checkout 固定到官方新版后，还要做一次很小的 Slurm cpuset 移植：

1. 将 Nano patch 中的 `resolve_cpu_affinity()` 移到 Edge
   `cosmos_framework/utils/device.py`。
2. 在 Edge `cosmos_framework/utils/distributed.py::init()` 中，只替换
   GPU-affinity 小块：把 NVML 请求与 `os.sched_getaffinity(0)` 求交集，
   并捕获 `pynvml.NVMLError` 和 `OSError`。
3. 保留 Edge 新增的 `COSMOS_DEVICE`、Gloo/NCCL 和 CPU checkpoint
   conversion 逻辑，不能用 Nano 文件整体覆盖。
4. 复用 `tests/test_device_cpu_affinity.py`，再跑一次 2/4 GPU Slurm
   smoke test；确认后为 Edge 重新生成独立 patch。

必要性和官方 `f734253` 代码证据见
`reports/cosmos3_piper14/2026-06-30_training_pitfalls_brief_zh.md` 第 8 节。

### 7.2 建议新增的文件

| 建议文件 | 从哪里参考 | 需要写什么 |
|---|---|---|
| `piper_cosmos/cosmos3/action_policy_piper14_edge.py` | 本仓库 `action_policy_piper14_nano.py` + 官方 `edge_model_config.py` | import `EDGE_MODEL_CONFIG`，注册 Piper，Edge generator/action optimizer allowlist |
| `configs/cosmos3/sft/action_policy_piper14_edge.toml` | 本仓库 Nano TOML | Edge DCP、4-GPU、480 tier、run 参数 |
| `piper_cosmos/cosmos3/train_action_policy_piper14_edge.py` | 当前 Nano train wrapper | 注册 Edge experiment；不要调用 Qwen-only bootstrap |
| `piper_cosmos/cosmos3/local_edge_assets.py` | `local_hf_assets.py` | 注册本地 Edge processor/reasoner snapshot 和 Wan VAE |
| `scripts/convert_cosmos3_edge_to_dcp_offline.py` | Nano conversion script | Edge HF 输入、Edge processor bootstrap、DCP 输出 |
| `scripts/run_convert_cosmos3_edge_dcp_slurm.sh` | Nano conversion Slurm | Edge 路径和验证报告 |
| `scripts/run_cosmos3_edge_piper14_sft_slurm.sh` | Nano train Slurm | Edge Python/TOML/checkpoint，4-GPU launch |
| `scripts/submit_cosmos3_edge_piper14_sft.sh` | Nano submit script | Edge readiness + sbatch |
| `scripts/cosmos3_edge_piper14_readiness.py` | 当前 readiness | 检查 Edge config、hidden size、domain collision、head skip、DCP metadata |

### 7.3 建议修改而不是复制的文件

`piper_cosmos/cosmos3/piper14_hdf5_action_dataset.py` 和 `domain.py` 的数据逻辑不用重写，但官方最新版把 namespace 从 `vfm` 移到 `generator`：

```text
旧：cosmos_framework.data.vfm.action...
新：cosmos_framework.data.generator.action...
```

可用 version-compatible import block 同时支持旧 Nano 和新 Edge，避免复制两份容易漂移的数据逻辑。必须用单元测试确认两个 namespace 下：

- raw sample shape 一致；
- domain ID 一致；
- SequencePlan condition mask 一致；
- resize bucket 一致；
- 14D padding/mask 一致。

`piper_cosmos/deployment/cosmos_piper14_policy.py` 需：

- 换新版 `data.generator` import；
- 允许 Edge checkpoint/config；
- 训练和推理同时启用 `format_prompt_as_json=True`；
- 确认服务端可以按 Piper domain 21 构建 batch；
- 保持只返回未来 32×14 action。

## 8. Edge experiment 的建议模型和 optimizer 配置

核心结构建议：

```python
edge_cfg = copy.deepcopy(EDGE_MODEL_CONFIG)
edge_cfg["action_gen"] = True
edge_cfg["vision_gen"] = True
edge_cfg["sound_gen"] = False
edge_cfg["resolution"] = "480"
edge_cfg["tokenizer"]["encode_exact_durations"] = [33]
edge_cfg["max_num_tokens_after_packing"] = -1
edge_cfg["rectified_flow_training_config"]["loss_scale"] = 10.0
edge_cfg["rectified_flow_training_config"]["action_loss_weight"] = 10.0
```

optimizer allowlist：

```python
keys_to_select = [
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "k_norm_und_for_gen",
    "action2llm",
    "llm2action",
    "action_modality_embed",
]
```

与 Nano 相比新增 `k_norm_und_for_gen`，因为官方 Edge vision SFT 在
`vision_sft_edge.py:L100-L114` 将它列为 Generator 路径的一部分。没有它会把 Edge 特有的 Reasoner-key normalization 错误冻结。

第一版 optimizer 建议优先沿用经过 action 训练验证的配置：

| 参数 | 建议 baseline | 依据 |
|---|---:|---|
| optimizer | FusedAdam | 官方 Nano action recipe 指出 BF16 action loss 对该设置更稳定 |
| betas | `(0.9, 0.99)` | Nano action recipe和当前 Piper run |
| eps | `1e-8` | 官方注释称 BF16 + `eps=1e-6` 曾导致 action loss divergence |
| weight decay | `0.05` | Nano action recipe和当前 Piper run |
| Generator nominal LR | `2e-4` | 控制变量：先复刻当前 Piper recipe |
| action nominal LR | `1e-3` | 5× |
| scheduler | LambdaLinear，20k cycle，`f_max=0.4 → 0` | 与 Nano 20k 做可比较 baseline |
| grad clip | `1.0` | action recipe已验证值 |

官方 Edge vision SFT 使用 AdamW、`lr=5e-4`、betas `(0.9,0.95)`、`eps=1e-6`、clip 0.1，但那是纯视觉 SFT，`action_gen=false`；不应直接替换 action recipe。它适合在 baseline 稳定后作为 optimizer ablation，而不是首跑默认值。

## 9. 数据配置：参考 DROID，但不照搬

官方 Nano DROID recipe 值得参考的部分：

- `policy` 单任务；
- current proprioceptive state；
- future action chunk；
- `ActionTransformPipeline`；
- episode-shuffle streaming；
- 480 tier；
- action channel mask；
- text CFG dropout 0.1；
- Generator + action head 联合训练；
- action loss 10、action LR 5×；
- train/inference prompt schema 完全一致。

不应复制的 DROID 参数：

| DROID 参数 | 为什么不能直接用到 Piper |
|---|---|
| 单臂 8D/10D joint action | Piper 是左右臂固定顺序的 14D absolute command |
| 15 FPS | Piper 数据确认是 30 FPS |
| DROID concat camera layout | Piper 是 high 在上、两个 wrist 在下 |
| DROID filter `keep_ranges_1_0_1.json` | trajectory key 和 idle 定义不适用于 Battery HDF5 |
| global batch 8192/HSDP 256 ranks | 这是官方大规模 DROID 基准，不适合当前 4-GPU/小数据设置 |
| DROID domain ID/action normalizer | Piper 必须保持 domain 21、raw absolute action |
| 单臂任务成功标准 | Battery Assemble 是双臂协同和真实装配 |

### Edge 的 joint action-mode 结果如何参考

Technical Report Appendix E.4（PDF 第 109 页）确实用 Cosmos3-Edge + PushT 比较了单独 FD/ID/policy 与联合 FD/ID/policy 训练；联合训练改善了 policy coverage 和 inverse-dynamics MSE。这证明 Edge 架构能承载 action 多任务，也说明后续联合训练有研究价值。

但它不是可直接复制到 Piper 的 recipe：

- PushT 是低维平面控制，Piper 是 14D 双臂真实机器人；
- 当前 `Piper14HDF5ActionDataset` 在 `L44-L49` 明确只允许 `mode="policy"`；
- FD 需要把 action 当 clean condition、预测 future video；ID 需要把完整 video 当 clean condition、预测 action，condition mask 与 policy 不同；
- 若要联合训练，必须新增明确的 mode sampler、每种 mode 的输入完整性检查、loss 统计和等预算对照。

所以首个 Edge-Piper baseline 仍保持 policy-only；FD/ID/policy 联合训练应作为 baseline 稳定后的独立实验，而不是为了“参考官方”直接打开一个字符串开关。

### prompt schema 建议

当前 Nano Piper run 用普通字符串 prompt，再追加 viewpoint/FPS/resolution。新版官方 DROID 和 Edge-DROID server 使用 JSON prompt，并明确要求训练、推理一致。

Edge 推荐默认：

```python
format_prompt_as_json=True
```

但 JSON 内容必须由 Piper 的真实 metadata 构造：14D action、`concat_view`、30 FPS、640×640 processing bucket、32-step horizon；不要硬编码 DROID 的单臂字段。为了判断改 prompt 是否本身带来变化，建议同时保留一个 plain-prompt 小规模对照，不要把 prompt schema 和 backbone 更换的影响混为一谈。

## 10. 分阶段训练方案

### Stage 0：静态和单样本 gate

必须全部通过后再占 4 GPU：

1. 新 framework 能实例化 `EDGE_MODEL_CONFIG`；
2. Edge HF → DCP 转换和 metadata 验证通过；
3. domain 21 没有冲突；
4. 单样本为 video `[3,33,720,640]`、action `[33,14]`；
5. transform 后视频为 480 tier 的 `640×640` bucket；
6. policy condition mask 为第 0 vision frame + 第 0 qpos row；
7. action pad 到 64D，但 loss mask 只覆盖 14D；
8. trainable parameter names 只包含 Generator、Edge `k_norm_und_for_gen` 和 action head；
9. Reasoner、Wan VAE、EMA 的 `requires_grad` 全为 false；
10. DCP load 日志明确显示 action head skipped/fresh initialized。

### Stage 1：4-GPU 100-step smoke

建议保持 Nano run 的 count-based global batch 上限 32：

```text
max_samples_per_batch=8
world_size=4
grad_accum_iter=1
global batch upper bound=32
```

通过标准：

- 100 step 无 NaN/Inf；
- action 和 vision loss 都非零且下降方向合理；
- domain 21 action head 有非零 gradient；
- Reasoner gradient 为无/0；
- save/load 一次 DCP 后输出可复现；
- 单 batch 推理得到有限的 `[32,14]` action。

### Stage 2：2k/5k pilot

- 每 1000 iter 保存；
- 固定离线 held-out episode，而不是只看 train loss；
- 比较 `iter_1000/2000/5000` 的 action MAE、速度/加速度平滑性、关节限位越界、短 horizon rollout；
- 与 Nano 20k 用相同数据、seed、preprocess 和推理参数对照。

### Stage 3：20k full run

只有 pilot 没有明显过拟合且真机 shadow/replay 安全检查通过，再跑 20k。当前 Nano run 关闭 validation；Edge 不建议继续这个盲点。至少应让
`scripts/cosmos3_piper14_offline_eval.py` 在固定 held-out episodes 上每个 checkpoint 运行，避免把“最后一个 checkpoint”默认当作“最好 checkpoint”。

## 11. 建议的 Edge baseline 参数

| 类别 | 建议值 |
|---|---|
| base checkpoint | `nvidia/Cosmos3-Edge` → 本地 `Cosmos3-Edge-DCP` |
| model | `EDGE_MODEL_CONFIG`，action/vision on，sound off |
| mode | `policy` |
| input | JSON instruction + current concat view + current 14D qpos |
| target | 32×14 future absolute action + auxiliary future video |
| FPS | 30 |
| frames | 33 |
| resolution | 480 tier；当前 Piper concat 对应 640×640 bucket |
| prompt | JSON，训练推理一致 |
| normalization | 第一版保持 `None`，与 Nano baseline 一致 |
| action loss | 10 |
| vision loss scale | 10 |
| CFG dropout | 0.1 |
| optimizer | FusedAdam `(0.9,0.99)`, eps `1e-8`, wd `0.05` |
| nominal LR | Generator `2e-4`；action `1e-3` |
| scheduler | LambdaLinear，20k，`f_max=0.4`，无 warmup（可比较 baseline） |
| precision | BF16 |
| parallel | 4-GPU FSDP shard 4 × replicate 1 |
| batch | max 8/rank，accum 1，global 上限 32 |
| grad clip | 1.0 |
| checkpoints | 每 500 iter |
| smoke/full | 100 → 2k/5k → 最多 20k |

这里的“建议 baseline”是为了与现有 Nano 实验做控制变量对比，不是声称这些值是 NVIDIA 已发布的 Edge-Piper 最优参数；官方目前没有 Edge-Piper recipe。

## 12. 推理参数和部署建议

### 12.1 第一版建议

直接从当前 Piper policy 参数起步：

| 参数 | 建议 |
|---|---:|
| `num_steps` | 4 |
| `guidance` | 3.0 |
| `shift` | 5.0 |
| `action_horizon` | 32 |
| `raw_action_dim` | 14 |
| `max_action_dim` | 64 |
| `fps` | 30 |
| `resolution` | `"480"` |
| `decode_video` | false |
| seed | 固定 0 做离线比较，真机按部署策略确定 |

本仓库对应代码：

- `piper_cosmos/deployment/cosmos_piper14_policy.py:L26-L45`
- 同文件 `L152-L180`
- `scripts/cosmos3_piper14_offline_eval.py:L38-L46`、`L263-L277`

官方 Edge-DROID 也证明 4-step PyTorch policy 是有效配置：其 real-time 测试使用 32×8 action、4 个 UniPC denoising steps、guidance 3.0、`conditioning_fps=15`。但官方另一套 vLLM 测试用 30 steps、guidance 1.0、5 Hz、320×192；两套 runtime/input protocol 不同，不能把 `30 steps + guidance 1` 与当前 PyTorch Piper 参数随意混搭。详见
[Edge-Policy-DROID model card](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID)。

### 12.2 必须实测的 sweep

固定同一 held-out 输入，至少比较：

```text
num_steps: 4, 8
guidance: 1.0, 2.0, 3.0
shift: 5.0（第一轮固定）
```

选择标准不是视频视觉质量，而是：

- action MAE/末端误差；
- action jerk 和相邻 chunk 不连续；
- 14D joint/gripper 合法范围；
- 单次延迟、P95 延迟；
- 在 RTC action consumption 策略下是否满足控制预算。

32 个 action 在 30 Hz 只覆盖约 1.07 秒。Edge 的“实时”宣传不能替代本机、当前 640×640 三相机拼图、当前 PyTorch 服务和 RTC 网络路径的端到端测量。

### 12.3 推理必须与训练一致

- 相机顺序和拼接：high 在上，left wrist 左下，right wrist 右下；
- current qpos 是 action 第 0 行；
- prompt JSON schema一致；
- domain 为 `piper14`/21；
- resize/pad 使用同一 `ActionTransformPipeline`；
- 输出只取未来 row 1–32、channel 0–13；
- 不做训练时不存在的 action normalization；
- 默认不解码未来视频以降低延迟。

## 13. 验收清单

### 代码/模型

- [ ] 使用支持 Edge 的固定 Cosmos Framework commit
- [ ] `Cosmos3-Edge-DCP` 转换和验证报告通过
- [ ] hidden size 2048，head shape 符合 Edge
- [ ] domain 21 无冲突
- [ ] fresh action head 的 skip log 可审计
- [ ] Reasoner/VAE 冻结，Generator/action 可训练

### 数据

- [ ] 14D 顺序与真机 controller 一致
- [ ] 30 FPS 来源确认
- [ ] 33 帧与 32 action horizon 一致
- [ ] 三相机时间同步
- [ ] held-out episode 不泄漏进训练
- [ ] prompt schema 训练推理一致

### 训练

- [ ] 100-step smoke 无 NaN
- [ ] action/vision loss 分开记录
- [ ] 2k/5k checkpoint 离线比较
- [ ] 与 Nano 20k 使用同一评估协议
- [ ] 不以 train loss 或最后 checkpoint 单独决定部署

### 真机

- [ ] 先 replay/shadow，不直接下发
- [ ] joint/gripper 限位和速度/加速度 clamp
- [ ] chunk 切换平滑/RTC 策略验证
- [ ] 小速度、软限位、人工急停的分阶段测试
- [ ] 记录端到端 P50/P95 latency 和真实成功率

## 14. 官方参考与采用边界

| 官方资料 | 本方案采用什么 | 不直接照搬什么 |
|---|---|---|
| [Cosmos3 Technical Report](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) | 双塔 MoT、冻结 Reasoner、联合 action/video diffusion、fresh action projection | DROID/PushT 的具体 embodiment 和数据分布 |
| [Nano action-policy fine-tuning cookbook](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/generator/action/finetune/README.md) | policy、state、chunk、stream shuffle、head 5× LR、DCP/FSDP | DROID 8/10D、15 FPS、filter、8192 global batch |
| [Cosmos Framework DROID post-training guide](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/action_policy_droid_posttrain.md) | HF→DCP、head fresh init、训练/推理 prompt 一致 | 文档中“base 可选 Edge”的一句泛化描述；现有可运行 experiment 仍是 Nano |
| [Edge model card](https://huggingface.co/nvidia/Cosmos3-Edge) | 4B、BF16、480p、action/video 能力、Edge base 初始化 | 官方未声明 Piper14 是原生支持 embodiment |
| [Edge-Policy-DROID model card](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID) | 4-step/guidance 3 的 PyTorch latency参考 | DROID 单臂 action、FPS、相机和控制协议 |
| [Edge policy server cookbook](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/generator/action/run_policy_with_cosmos_framework.md) | Edge checkpoint 和 JSON prompt 的服务方式 | RoboLab client/domain |
| [Edge vision SFT config](https://github.com/NVIDIA/cosmos-framework/blob/main/cosmos_framework/configs/base/experiment/sft/vision_sft_edge.py) | `EDGE_MODEL_CONFIG`、`k_norm_und_for_gen`、4-GPU 可行性参考 | 纯视觉 optimizer 直接用于 action SFT |

最终建议是先完成一个**可审计、与 Nano 20k 可对照的 Edge-base + fresh Piper head baseline**，再分别做 optimizer、prompt 和 Edge-DROID transfer ablation。这样任何性能变化都能定位原因，而不是一次同时改 backbone、action 语义、prompt、FPS 和训练规模。

## 15. 基于 master 创建 Edge 分支后的实施计划

### 15.1 分支起点和边界

外层不要重新 clone。先刷新远端，再用现有仓库直接创建 Edge branch 和独立
工作目录：

```bash
cd /project/peilab/wam/cosmos3_cy
git fetch origin

git worktree add \
  /project/peilab/wam/cosmos3_cy_edge \
  -b cosmos3-edge-piper14 \
  origin/master

git -C /project/peilab/wam/cosmos3_cy_edge \
  push -u origin cosmos3-edge-piper14
```

执行后，`cosmos3_cy/` 固定用于 Nano，`cosmos3_cy_edge/` 固定用于
Edge；两个目录共享外层 Git objects/history，但各自有独立工作文件，因此
后续不需要在同一目录来回 `git switch`。

新分支会继承本项目已提交的 Piper14 代码、配置、测试和 patch，但不会自动
复制以下内容：

- 外层工作区尚未提交的修改；
- `external/cosmos` 的当前工作状态；
- 被 `external/cosmos/.gitignore` 忽略的 `packages/cosmos3`；
- checkpoint、venv、cache 和训练输出。

因此，Edge 工作区还要一次性建立独立的 cookbook/framework checkout，并把
两者 commit 写入 Edge dependency lock。优先从现有本地仓库创建 Git
worktree；只拉取缺失对象，不必重新下载完整 Git 历史。

具体顺序是：

1. 在现有 `external/cosmos` 中 fetch 并审计支持 Edge 的 cookbook
   commit，记为 `<EDGE_COSMOS_COMMIT>`。
2. 从现有 cookbook Git 仓库为
   `/project/peilab/wam/cosmos3_cy_edge/external/cosmos`
   建立独立 worktree，并固定到 `<EDGE_COSMOS_COMMIT>`。
3. 在现有 `external/cosmos/packages/cosmos3` 中 fetch 官方
   `cosmos-framework`，把经审计的 Edge commit 记为
   `<EDGE_FRAMEWORK_COMMIT>`。
4. 从现有 framework Git 仓库为 Edge 路径建立独立 worktree，并同时创建
   本地修复分支：

   ```bash
   git -C /project/peilab/wam/cosmos3_cy/external/cosmos/packages/cosmos3 \
     worktree add \
     -b edge-piper14-cpuset \
     /project/peilab/wam/cosmos3_cy_edge/external/cosmos/packages/cosmos3 \
     <EDGE_FRAMEWORK_COMMIT>
   ```

5. 在 `edge-piper14-cpuset` 中完成 cpuset 小型移植、测试和提交，再把
   `format-patch` 保存到 Edge 外层分支的
   `patches/cosmos-framework/`。

外层仓库没有可用的 `.gitmodules` 自动初始化链路，所以不能把以上步骤替换
成一次外层 `git pull`。执行嵌套 worktree 命令前，应确认目标 gitlink
placeholder 为空且目标路径没有用户文件；若不满足，先检查目录内容，不要
直接强制覆盖。

checkpoint 也不会因为 worktree 自动共享。Nano 20k checkpoint 保持在原
路径；Edge 使用官方 Edge HF checkpoint 和转换后的 Edge DCP。若要避免重复
大文件，应把不可变模型资产放到明确的共享绝对路径，并在
`configs/dependencies/cosmos3_edge.lock.yaml` 记录真实路径和校验值，而不是
把 checkpoint 提交进 Git。

### 15.2 当前已有、需要适配和缺失项

| 状态 | 内容 | 文件或目标 |
|---|---|---|
| 已有，可直接复用 | Piper14 domain、14D action 语义 | `piper_cosmos/cosmos3/domain.py` |
| 已有，可直接复用 | Battery HDF5、30 FPS、三相机 `concat_view`、policy chunk | `piper_cosmos/cosmos3/piper14_hdf5_action_dataset.py` |
| 已有，可作为基线 | Generator + fresh action head、action 5× LR、20k schedule | `piper_cosmos/cosmos3/action_policy_piper14_nano.py` |
| 已有，可作为骨架 | Slurm、readiness、preflight、DCP 验证 | `scripts/*cosmos3_piper14*`、`scripts/verify_cosmos3_dcp.py` |
| 已有，可作为骨架 | policy server、RTC、14D 输出裁剪 | `piper_cosmos/deployment/cosmos_piper14_policy.py` 等 |
| 需要适配 | 旧 `data.vfm` import | 改为新版 `data.generator`，或增加受测兼容层 |
| 需要适配 | Nano/Qwen 本地资产 bootstrap | 改为 Edge processor/reasoner 资产解析 |
| 需要适配 | plain prompt 和旧部署 import | 训练、推理统一 Edge JSON prompt |
| 缺失 | Edge framework 工作目录 | 固定官方 `f734253` 或后续经审计的明确 commit |
| 缺失 | Edge HF checkpoint 和 DCP | `Cosmos3-Edge`、`Cosmos3-Edge-DCP` |
| 缺失 | Edge experiment/config/train wrapper | 见 15.3 |
| 缺失 | Edge conversion/submit/readiness | 见 15.3 |
| 缺失 | Edge cpuset 修复提交和 patch | 从 Nano patch 做两处小型移植 |
| 缺失 | Edge smoke、pilot、离线和真机结果 | 按 15.4 gate 产生 |

“已有”不表示 Nano 文件可以原样执行 Edge。尤其 Nano hidden size 为 4096，
Edge 为 2048；Piper action-head 结构和 14D 语义可以复用，但 Nano 20k 的
投影权重不能直接加载到 Edge，第一版使用 fresh domain-21 head。

### 15.3 必须落地的文件

第一批新增：

```text
configs/dependencies/cosmos3_edge.lock.yaml
piper_cosmos/cosmos3/action_policy_piper14_edge.py
piper_cosmos/cosmos3/train_action_policy_piper14_edge.py
piper_cosmos/cosmos3/local_edge_assets.py
configs/cosmos3/sft/action_policy_piper14_edge.toml
scripts/convert_cosmos3_edge_to_dcp_offline.py
scripts/run_convert_cosmos3_edge_dcp_slurm.sh
scripts/submit_convert_cosmos3_edge_dcp.sh
scripts/run_cosmos3_edge_piper14_sft_slurm.sh
scripts/submit_cosmos3_edge_piper14_sft.sh
scripts/cosmos3_edge_piper14_readiness.py
```

第一批修改：

```text
piper_cosmos/cosmos3/domain.py
piper_cosmos/cosmos3/piper14_hdf5_action_dataset.py
piper_cosmos/deployment/cosmos_piper14_policy.py
```

Edge framework 内部修改：

```text
cosmos_framework/utils/device.py
cosmos_framework/utils/distributed.py
cosmos_framework/utils/distributed_test.py
```

framework 修改必须在第三层独立分支提交，并在外层
`patches/cosmos-framework/` 保存 Edge 专用 `format-patch`。不能让 Edge
framework 只保持为 dirty working tree。

### 15.4 按依赖排序的执行阶段

1. **工作区固定**
   - 创建外层 `cosmos3-edge-piper14` worktree；
   - 建立独立 cookbook/framework checkout；
   - 固定 URL、commit、Python/CUDA/PyTorch 和资产路径。
2. **Framework gate**
   - 验证 `EDGE_MODEL_CONFIG`、Nemotron Dense VL、SigLIP2 和新 namespace；
   - 移植 cpuset 修复；
   - 保留官方 Gloo/NCCL 与 CPU checkpoint conversion；
   - 通过 affinity 和官方 distributed 单测。
3. **资产 gate**
   - 下载并校验官方 `Cosmos3-Edge`；
   - 用同一固定 framework 转换为 DCP；
   - 保存转换命令、metadata、文件清单和校验报告。
4. **Piper integration gate**
   - 实现 Edge experiment、TOML、train wrapper 和本地资产 helper；
   - 验证 domain 21、14D 顺序、33 帧/32 action、480 tier 和 JSON prompt；
   - 验证 Reasoner/VAE 冻结，Generator/action head 可训练；
   - 验证 base load 明确跳过 fresh Piper head。
5. **训练 gate**
   - 单样本 forward/backward；
   - 4-GPU 100-step smoke；
   - 2k/5k pilot 和 held-out 离线比较；
   - gate 全部通过后才启动 20k。
6. **部署 gate**
   - 离线 sweep `steps={4,8}`、`guidance={1,2,3}`；
   - replay/shadow、限位和 RTC 延迟测试；
   - 最后才进入低速真机闭环。

### 15.5 “可以开始训练”的最低完成定义

只有同时满足以下条件，才把 Edge 分支标记为可训练：

- dependency lock 能唯一还原三层代码和运行环境；
- Edge HF → DCP 转换可重复，readiness 报告通过；
- dataset 单测证明相机顺序、FPS、horizon、14D 和 mask 未漂移；
- checkpoint load 日志证明 fresh Piper head，没有静默 shape mismatch；
- trainable parameter 清单只包含预期 Generator、Edge
  `k_norm_und_for_gen` 和 action head；
- cpuset、单卡单样本和 4-GPU 100-step smoke 全部通过；
- 输出目录、W&B、checkpoint interval 和 resume 行为已经验证。

在这些条件完成前，分支状态应标为“Edge 接入中”，不能因为
`EDGE_MODEL_CONFIG` 能 import 就认为后训练链路已经打通。
