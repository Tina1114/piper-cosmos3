# Cosmos Piper14 checkpoint 量化框架

## 环境
```bash
cd /root/piper-cosmos3-quant
source /root/cosmos3/cosmos/packages/cosmos3/.venv/bin/activate
export REPO_ROOT="$PWD"
export COSMOS_FRAMEWORK_ROOT="$PWD/external/cosmos/packages/cosmos3"
export COSMOS_PIPER14_PYTHON="$COSMOS_FRAMEWORK_ROOT/.venv/bin/python"
export COSMOS_PYTHON="$PWD/external/cosmos/packages/cosmos3/.venv/bin/python"
export PYTHONPATH="$REPO_ROOT:$COSMOS_FRAMEWORK_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME="$PWD/external/cosmos/checkpoints/hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export COSMOS3_QWEN_SNAPSHOT="$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
export COSMOS3_WAN_VAE_PATH="$HF_HOME/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# 4090 加载完整模型需要 Reasoner offload；
# 完成 prefill 后 GEN 常驻 GPU，3.3 秒 denoise 阶段没有逐层 offload。
export COSMOS3_REASONER_OFFLOAD=1

# 对应约 55~64ms 的稳态数据/VAE prepare。
export COSMOS3_VAE_GPU_RESIDENT=1
unset COSMOS3_VAE_CPU_OFFLOAD
unset COSMOS3_LAYER_OFFLOAD
unset COSMOS3_REASONER_EMPTY_CACHE
unset COSMOS3_REASONER_KVCACHE_DEBUG

mkdir -p logs/deployment

```


## 目标

量化被拆成两个独立阶段：

```text
原始 HF checkpoint
  -> create checkpoint（algo + backend）
  -> 持久化量化 checkpoint + quantization_config.json

量化 checkpoint
  -> server 启动时校验 manifest
  -> backend 准备 runtime
  -> 原有 Cosmos ActionModelService
  -> policy inference
```

当前内置组合：

```text
algo:    rtn（默认）
backend: fakequant（默认）
```

`RTN + fakequant` 在创建阶段对权重做 symmetric round-to-nearest，然后反量化回原始
dtype，并保存为标准 Hugging Face sharded safetensors。开启 activation quantization 后，
加载阶段还会把 manifest 选中的 `nn.Linear` 替换成 `W4A8FakeQuantLinear`：权重使用
output-channel-wise W4，输入在每次 forward 动态执行 token-wise A8。

这个 backend 用于建立数值精度基线。checkpoint 仍是 BF16/FP32 大小；运行时的 W4 数值存放
在 int8 tensor 中（没有做 4-bit packing），矩阵乘法前会反量化并调用普通浮点 Linear，
所以不应期待真实 W4A8 kernel 的速度。

后续真实 INT4/INT8 backend 可以把 `qweight/scale/zero_point` 持久化，并在
`prepare_runtime()` 中准备 checkpoint、在 `prepare_model()` 中安装对应的 module
replacement/kernel loader。

## 代码结构

```text
piper_cosmos/quantization/
  spec.py                 # QuantizationSpec：algo/backend/bits/group/scope
  registry.py             # 算法和 backend 注册表、公共协议
  algorithms/rtn.py       # RTN quantize/dequantize
  backends/fakequant.py   # 标准 safetensors fake-quant backend
  fakequant_linear.py     # output-channel W4 + token-wise A8 Linear
  checkpoint.py           # checkpoint plan/create/原子发布
  manifest.py             # quantization_config.json schema
  runtime.py              # 推理前校验与 backend runtime 准备

scripts/
  create_cosmos_piper14_quant_checkpoint.py
  serve_cosmos_piper14_policy.py
```

## 1. 预检查

`--dry-run` 只读取 safetensors header，不读取全部权重：

```bash
cd /root/piper-cosmos3-quant

COSMOS_PYTHON="$PWD/external/cosmos/packages/cosmos3/.venv/bin/python"
export PYTHONPATH="$PWD:$PWD/external/cosmos/packages/cosmos3"

"$COSMOS_PYTHON" scripts/create_cosmos_piper14_quant_checkpoint.py \
  --checkpoint cosmos_battery/20k \
  --output cosmos_battery/20k-rtn-w8-fakequant \
  --algo rtn \
  --backend fakequant \
  --bits 8 \
  --group-size 128 \
  --scope all \
  --dry-run
```

当前 20k checkpoint 的 dry-run 结果：

```text
shards:                 7
tensors:                1160
selected_tensors:       632
source_tensor_bytes:    31,499,049,824
selected_tensor_bytes:  31,496,859,648
estimated output:       与原 checkpoint 基本相同（fakequant）
```

默认只量化浮点且 `ndim >= 2` 的权重；bias、标量、整数 tensor 会原样保存。

## 2. 创建并保存 checkpoint

### Weight-only 示例

```bash
"$COSMOS_PYTHON" scripts/create_cosmos_piper14_quant_checkpoint.py \
  --checkpoint cosmos_battery/20k \
  --output cosmos_battery/20k-rtn-w8-fakequant \
  --algo rtn \
  --backend fakequant \
  --bits 8 \
  --group-size 128 \
  --scope all
```

### W4A8 示例

```bash
cd /root/piper-cosmos3-quant

export COSMOS_PYTHON="$PWD/external/cosmos/packages/cosmos3/.venv/bin/python"
export PYTHONPATH="$PWD:$PWD/external/cosmos/packages/cosmos3"

"$COSMOS_PYTHON" scripts/create_cosmos_piper14_quant_checkpoint.py \
  --checkpoint cosmos_battery/20k \
  --output cosmos_battery/20k-rtn-w4a8-gen-fakequant \
  --algo rtn \
  --backend fakequant \
  --weight-bits 4 \
  --activation-bits 8 \
  --weight-granularity output_channel \
  --activation-granularity token \
  --group-size 0 \
  --scope generator
```

```bash
"$COSMOS_PYTHON" scripts/create_cosmos_piper14_quant_checkpoint.py \
  --checkpoint "$PWD/cosmos_battery/20k" \
  --output "$PWD/cosmos_battery/20k-rtn-w4a8-gen-cutlass" \
  --algo rtn \
  --backend cutlass \
  --weight-bits 4 \
  --activation-bits 8 \
  --weight-granularity output_channel \
  --activation-granularity token \
  --group-size 0 \
  --scope generator

```

`group-size=0 + weight-granularity=output_channel` 表示每个 `[out_channels,in_channels]`
权重矩阵的每个输出行独立计算一个 W4 scale。激活在 create 阶段不会被量化；manifest 只记录
A8 配置，真正的 token-wise 动态量化发生在推理 forward。

创建过程按 shard 读取和写入。目标目录不能已经存在，工具不会覆盖已有 checkpoint。
发布前先写入同级临时目录，全部 shard 和 manifest 成功后才原子 rename 到目标路径。

输出保留：

```text
config.json
checkpoint.json
model.safetensors.index.json
model-00001-of-00007.safetensors
...
model-00007-of-00007.safetensors
quantization_config.json
```

### 选择量化范围

只量化 diffusion/GEN expert：

```bash
--scope generator
```

只量化 Language/Reasoner UND：

```bash
--scope reasoner
```

用 tensor name 进一步过滤：

```bash
--include-regex 'language_model\.model\.layers' \
--exclude-regex 'embed_tokens|lm_head'
```

多个 `--include-regex` 是 OR；命中任意 `--exclude-regex` 都会跳过。

## 3. 加载量化 checkpoint 并推理

环境配置与原 policy server 相同，只替换 checkpoint，并声明 manifest 中预期的
`algo/backend`：

```bash
"$COSMOS_PYTHON" scripts/serve_cosmos_piper14_policy.py \
  --checkpoint "$PWD/cosmos_battery/20k-rtn-w4a8-gen-cutlass" \
  --config-file "$PWD/configs/cosmos_piper14_20k_local_config.json" \
  --algo rtn \
  --backend cutlass \
  --require-quantized-checkpoint \
  --host 0.0.0.0 \
  --port 8766 \
  --condition-only-vae \
  --instruction-cache \
  --action-horizon 32 \
  --num-steps 4 \
  --guidance 3.0 \
  --shift 5.0
```

模型加载前应看到：

```text
[cosmos-piper14-quantization] checkpoint=... algo=rtn backend=fakequant ...
[cosmos-piper14-quantization-runtime] module_type=W4A8FakeQuantLinear \
replaced_linear_modules=... activation_quantization=True
```

以下情况会在加载 30GB 模型之前直接失败：

- 缺少 `quantization_config.json`，但传了 `--require-quantized-checkpoint`。
- CLI `--algo` 和 manifest 不一致。
- CLI `--backend` 和 manifest 不一致。
- manifest schema 或 storage format 不受支持。

推理 client 不需要改动：

```bash
cd /root/piper-cosmos3-quant

export COSMOS_FRAMEWORK_ROOT="$PWD/external/cosmos/packages/cosmos3"
export COSMOS_PIPER14_PYTHON="$COSMOS_FRAMEWORK_ROOT/.venv/bin/python"
export PYTHONPATH="$PWD:$COSMOS_FRAMEWORK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export COSMOS_PIPER14_CLIENT_TIMING=1

mkdir -p reports/cosmos3_piper14

"$COSMOS_PIPER14_PYTHON" scripts/dry_run_piper14_rtc_runtime.py \
  --host 127.0.0.1 \
  --port 8766 \
  --authkey cosmos-piper14 \
  --data-root "$PWD/hdf_data/perfect" \
  --data-config "$PWD/configs/dataset_configs/battery_assemble_hdf5.yaml" \
  --prompt "Assemble the mouse's battery." \
  --steps 384 \
  --chunk-size 32 \
  --replan-interval 32 \
  --control-hz 30 \
  --loop \
  --timing \
  --report reports/cosmos3_piper14/profile_4step_3p3s.json
```

Server metadata 中会增加：

```text
{
  "quantization": {
    "active": true,
    "algo": "rtn",
    "backend": "fakequant",
    "bits": 4,
    "activation_bits": 8,
    "group_size": 0,
    "weight_granularity": "output_channel",
    "activation_granularity": "token",
    "scope": "all",
    "storage_format": "hf-safetensors-fakequant-v1",
    "module_type": "W4A8FakeQuantLinear",
    "replaced_linear_modules": <runtime count>,
    "activation_quantization": true
  }
}
```

## 扩展新算法

实现 `QuantizationAlgorithm`：

```python
class GPTQAlgorithm:
    name = "gptq"

    def quantize(self, tensor, spec):
        ...

    def dequantize(self, value, spec):
        ...

register_algorithm(GPTQAlgorithm())
```

然后在 `registry._ensure_builtins()` 中导入模块，CLI 会自动把它加入 `--algo` choices。
需要 calibration 数据的算法可以扩展 create CLI，先收集 Hessian/statistics，再调用同一
checkpoint writer。

## 扩展真实 kernel backend

实现 `QuantizationBackend`：

```python
class TorchAOBackend:
    name = "torchao"
    storage_format = "cosmos-packed-torchao-v1"

    def encode_tensor(self, name, tensor, algorithm, spec):
        return {
            f"{name}.qweight": ...,
            f"{name}.scale": ...,
        }

    def prepare_runtime(self, checkpoint, manifest):
        return checkpoint

    def prepare_model(self, model, manifest, algorithm):
        # 模型加载完成后注册 module replacement/kernel。
        return {"replaced_linear_modules": ...}

register_backend(TorchAOBackend())
```

真实 packed backend 还需要让 Cosmos loader 理解新增的 tensor names。应在 backend 内完成
module replacement 或 state-dict hook，不要把 kernel 特定逻辑写入 Piper policy wrapper。
