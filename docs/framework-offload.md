# Framework offload 策略(piper 侧说明)

> piper14 推理在 **4090 (24GB)** 上跑 Cosmos3-Nano(~23GB)必须 offload。offload 改在 **cosmos-framework 源码**里(env 门控),不是 piper 代码。这份说明记录:piper 怎么把这套 offload 落到它依赖的 framework 克隆上。

## 1. offload 是什么(两段,都 env 门控,默认关)

| 策略 | env 开关 | 作用 | 落点(framework 文件) |
|---|---|---|---|
| **VAE CPU offload** | `COSMOS3_VAE_CPU_OFFLOAD=1` | VAE build 在 CPU、encode/decode 时搬 GPU 用完搬回;跳过 CPU 上的 `sync_model_states` | `model/generator/tokenizers/wan2pt2_vae_4x16x16.py` |
| **DiT 逐层流式** | `COSMOS3_LAYER_OFFLOAD=1` | 关 torch.compile、DiT materialize 到 CPU、非 layer 小件钉 GPU、block 循环每层 H2D→算→D2H | `model/generator/omni_mot_model.py`(`build_net`)、`model/generator/mot/unified_mot.py`(block 循环) |

两个开关都关时,framework 行为与上游完全一致(训练/其他路径零影响)。

## 2. 为什么是 framework 改动,却要在 piper 体现

piper 经 `external/cosmos/packages/cosmos3` 消费 framework(`prepare_cosmos3_fd_env.sh` clone cosmos-framework)。offload 改的是 framework 源码,属于 **framework fork** 的事;但 piper 要能**复现**这套 offload,所以把 framework diff 以 **patch** 形式随 piper 仓库走:

```
patches/framework-offload.patch   # framework offload 的 git diff(VAE offload + 逐层流式,3 文件)
```

## 3. 怎么应用(在 piper 环境)

`prepare_cosmos3_fd_env.sh` clone 完 framework 后,apply 这个 patch:

```bash
cd ~/piper-cosmos3
FW=external/cosmos/packages/cosmos3          # framework 克隆根
# (re)apply offload
(cd "$FW" && git apply --whitespace=nowarn "$PWD/../../patches/framework-offload.patch" 2>/dev/null \
   || cd "$FW" && git apply /root/piper-cosmos3/patches/framework-offload.patch)
# 验证
grep -R "COSMOS3_LAYER_OFFLOAD\|COSMOS3_VAE_CPU_OFFLOAD" "$FW/cosmos_framework" | head
```

> 重新 clone / `git pull` framework 后**要重新 apply**(patch 会被打回)。
> 理想做法是维护一个 **framework fork 分支**带这套 offload,piper 的 `external/cosmos` 指向它;patch 是便携形式。

## 4. 启用 offload 跑推理/评估

serve:
```bash
COSMOS3_LAYER_OFFLOAD=1 COSMOS3_VAE_CPU_OFFLOAD=1 \
HF_HOME=~/piper-cosmos3/external/cosmos/checkpoints/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=~/piper-cosmos3:~/piper-cosmos3/external/cosmos/packages/cosmos3 \
~/piper-cosmos3/external/cosmos/packages/cosmos3/.venv/bin/python \
  scripts/serve_cosmos_piper14_policy.py --checkpoint cosmos_battery/20k --host 0.0.0.0 --port 8766
```

offline_eval:同上 env + `scripts/cosmos3_piper14_offline_eval.py ...`(详见 `docs/offload/eval-report-2026-07-14.md`,或在 cosmos3 仓库)。

## 5. 实测(2026-07-14,4090,20k checkpoint,2 episode)

- model offload 加载 214.78s;去噪 32 步跑完(07:13,~13.5s/it);**GPU 峰值 ~6GB/24GB**。
- `offline_eval` `status: passed`,`pred_action [1,32,14]` finite,`per_frame_mae=0.1358`。
- → **offload 不破坏数值正确性**。latency ~7min/样本(同步流式 + 每步流式 und+gen),提速待 anchor F(und prefill skip + async 双流)。

## 6. 相关

- patch:`patches/framework-offload.patch`
- 设计/锚点:cosmos3 仓库 `docs/offload/inference-offload-anchors.md`
- 改动记录:cosmos3 仓库 `docs/offload/CHANGELOG.md`
- 测试报告:cosmos3 仓库 `docs/offload/eval-report-2026-07-14.md`
