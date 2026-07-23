# W4A8 CUTLASS fused kernel

## 当前实现边界

当前 `backend=cutlass` 已经实现：

```text
checkpoint:
  symmetric RTN W4
  output-channel scale
  UINT8 packed qweight（low nibble first）

runtime:
  BF16 activation
    -> CUDA token-wise A8
    -> CUTLASS global INT4 iterator
    -> register INT4 unpack/sign extend
    -> shared-memory INT8 B tile
    -> INT8 Tensor Core GEMM
    -> INT32 accumulator
    -> scale/bias epilogue
    -> BF16 output
```

这是真实的 fused W4A8 Tensor Core kernel，不会调用 `torch.nn.functional.linear`，不会
静默退回 fakequant，也不会创建全局 INT8 weight 临时 tensor。

CUTLASS `MmaPipelined` 的 global B iterator 直接读取 `int4b_t`，随后使用
`NumericArrayConverter<int8_t,int4b_t>` 在寄存器中展开，再写入 INT8 shared-memory
tile。warp MMA 保持 `s8 × s8 → s32`。

旧的“CUDA 全局 unpack + CUTLASS W8A8”实现仍通过
`torch.ops.piper_w4a8.linear_debug` 保留，只作为逐元素正确性 oracle。

第一版只支持：

- NVIDIA SM89；
- BF16 activation/output；
- W4 output-channel-wise symmetric RTN；
- A8 token-wise symmetric dynamic quantization；
- 偶数 K，CUTLASS 路径要求 `K % 16 == 0`；
- inference only。

zero point 字段暂不保存；对称路径等价于 `zero_point=0`。

## 1. 构建扩展

当前 PyTorch 是 `2.10.0+cu130`，必须使用 CUDA 13.0 的 nvcc：

```bash
cd /root/piper-cosmos3-quant

COSMOS_PYTHON="$PWD/external/cosmos/packages/cosmos3/.venv/bin/python"
export CUDA_HOME=/opt/cuda-13.0-vllm
export PATH="$CUDA_HOME/bin:$PATH"
export PIPER_CUTLASS_DIR=/root/qi/third_party/cutlass
export TORCH_CUDA_ARCH_LIST=8.9

"$COSMOS_PYTHON" setup_quant_kernels.py build_ext --inplace --force
```

不要用 `/usr/local/cuda`：它目前是 CUDA 12.8，与 cu130 PyTorch 的 CUDA major 不一致。
当前测试使用 CUTLASS v4.4.2。

构建前检查：

```bash
command -v nvcc
readlink -f "$(command -v nvcc)"
nvcc --version

"$COSMOS_PYTHON" - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA_HOME:", CUDA_HOME)
print("GPU:", torch.cuda.get_device_name(0))
print("compute capability:", torch.cuda.get_device_capability(0))
PY
```

正确结果应包含：

```text
nvcc:               /opt/cuda-13.0-vllm/bin/nvcc
torch:              2.10.0+cu130
torch CUDA:         13.0
compute capability: (8, 9)
```

验证：

```bash
"$COSMOS_PYTHON" -m unittest discover \
  -s tests -p 'test_w4a8_cuda.py' -v
```

## 2. 创建 packed checkpoint

首版 sidecar 只量化 Generator 的 252 个 Linear：

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

每个被选择的：

```text
<module>.weight
```

会变成：

```text
<module>.qweight       UINT8 [N,K/2]
<module>.weight_scale FP32 [N]
```

未选择的 tensor 保持原样。manifest 记录原始 checkpoint 路径。

## 3. 加载并推理

### 3.1 推荐环境

当前 24GB RTX 4090 的正确配置与 `docs/quantization.md` 一致：

```bash
cd /root/piper-cosmos3-quant

export REPO_ROOT="$PWD"
export COSMOS_FRAMEWORK_ROOT="$PWD/external/cosmos/packages/cosmos3"
export COSMOS_PIPER14_PYTHON="$COSMOS_FRAMEWORK_ROOT/.venv/bin/python"
export COSMOS_PYTHON="$COSMOS_PIPER14_PYTHON"
export PYTHONPATH="$REPO_ROOT:$COSMOS_FRAMEWORK_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME="$PWD/external/cosmos/checkpoints/hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export CUDA_HOME=/opt/cuda-13.0-vllm
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# 必须打开：UND/Reasoner decoder 层直接加载到 CPU；
# Generator 在量化替换前以 BF16 常驻 GPU，替换后以 packed W4 常驻 GPU。
export COSMOS3_REASONER_OFFLOAD=1

# 量化后有足够空间让 VAE 常驻 GPU，不需要 VAE CPU offload。
export COSMOS3_VAE_GPU_RESIDENT=1
unset COSMOS3_VAE_CPU_OFFLOAD

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset COSMOS3_LAYER_OFFLOAD
unset COSMOS3_REASONER_EMPTY_CACHE
unset COSMOS3_REASONER_KVCACHE_DEBUG
```

`COSMOS3_REASONER_OFFLOAD=1` 与通用的 `COSMOS3_LAYER_OFFLOAD` 不同：当前配置只把 UND
decoder layer 放在 CPU，Generator 始终常驻 GPU。不要同时设置两者。

### 3.2 Instruction cache 的实际行为

`--instruction-cache` 是进程内的 UND K/V memory cache，不是“启动时不加载 Reasoner
权重”：

| 阶段 | Reasoner/UND | Generator |
|---|---|---|
| 模型启动 | 36 层 UND 权重加载到 CPU；UND bridge 留在 GPU | 先加载 BF16，随后替换为 packed W4 |
| 第一次相同 prompt（cache miss） | conditional/unconditional 各执行一次 prefill，逐层流式到 GPU | 执行去噪 |
| 后续相同 prompt（cache hit） | 复用 `gen_only` memory，不再执行 UND prefill | 只执行 Generator 去噪 |

因此 cache hit 后的稳态去噪确实只计算生成塔，但当前服务启动和第一次请求仍然需要
Reasoner 权重。服务重启后，进程内 cache 也需要重新建立。

### 3.3 启动命令

```bash
"$COSMOS_PIPER14_PYTHON" scripts/serve_cosmos_piper14_policy.py \
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

sidecar 流程会先从 manifest 指向的原始 checkpoint 加载 Cosmos，然后从 packed artifact
逐 shard 替换 Generator Linear。日志中应出现：

```text
module_type=W4A8CutlassLinear
replaced_linear_modules=252
kernel=cutlass-w4a8-fused-mainloop
```

### 3.4 量化/未量化的权重加载显存峰值

以下结果使用同一台 RTX 4090 和相同 placement：

```text
COSMOS3_REASONER_OFFLOAD=1
COSMOS3_VAE_GPU_RESIDENT=1
instruction cache enabled
```

| checkpoint | `nvidia-smi` 启动/权重加载峰值 |
|---|---:|
| 原始 BF16 20k | 17,950 MiB（17.53 GiB） |
| W4A8 CUTLASS sidecar | 17,956 MiB（17.54 GiB） |

两者加载峰值基本相同，因为当前 CUTLASS backend 仍然是 sidecar：

```text
加载原 BF16 Generator
  -> 模型加载完成
  -> 逐层释放 BF16 Linear
  -> 安装 packed W4 Linear
```

checkpoint 统计：

| 项目 | 大小 |
|---|---:|
| 原 Generator BF16 Linear | 12.9375 GiB |
| packed W4 + FP32 scales | 3.2396 GiB |
| 替换后理论减少的 PyTorch allocated | 9.6979 GiB |

替换后的 BF16 storage 会回到 PyTorch caching allocator，因此同一进程可以在推理时复用
这约 9.70 GiB；但 allocator 可能继续把它标为 `reserved`，所以替换完成后
`nvidia-smi` 不会立刻下降。不要只用 `nvidia-smi` 的进程占用判断 packed 权重是否生效，
还应检查 `replaced_linear_modules=252` 和 PyTorch `memory_allocated`。

关闭 `COSMOS3_REASONER_OFFLOAD` 的实测结果是在 `net.to_empty(cuda)` 阶段 OOM：

```text
PyTorch allocated: 23.07 GiB
进程显存:         23.45 GiB
下一次分配:       96 MiB
```

这个 OOM 发生在 packed module replacement 之前，instruction cache 和 allocator
`expandable_segments` 都无法避免。只有实现 standalone packed checkpoint 的 pre-load
hook，让 Cosmos 从一开始就不分配 BF16 `_moe_gen.weight`，才可以重新评估完全关闭
Reasoner offload。

当前版本仍需要 manifest 指向的原始 checkpoint。后续 standalone 版本需要在 Cosmos
loader 之前替换 module structure，直接跳过原始 `_moe_gen.weight`。

## 4. Microbenchmark

```bash
"$COSMOS_PYTHON" scripts/benchmark_w4a8_cutlass.py \
  --tokens 3233 \
  --warmup 10 \
  --iterations 50
```

该脚本覆盖 `q/o`、`k/v`、`gate/up`、`down` 四组真实矩阵形状，并输出单次 kernel 时间
及 effective TOPS。

## 5. 当前 4090 microbenchmark

`M=3233` 时，fused mainloop 相对旧 debug 路径：

| Linear | fused | debug unpack | 提升 |
|---|---:|---:|---:|
| q/o | 0.253 ms | 0.292 ms | 1.16x |
| k/v | 0.085 ms | 0.089 ms | 1.05x |
| gate/up | 0.690 ms | 0.775 ms | 1.12x |
| down | 0.837 ms | 0.902 ms | 1.08x |

这些是单 Linear microbenchmark，不代表完整请求耗时。

## 6. 下一阶段

下一版保持 Python/checkpoint 接口不变，继续增加：

```text
按 shape dispatch 64x128 / 64x256 / 128x64 tile
QKV 共用一次 activation quant
gate/up 共用一次 activation quant
standalone packed checkpoint pre-load hook
```

fused mainloop 会继续与 `linear_debug` 逐层对齐。
