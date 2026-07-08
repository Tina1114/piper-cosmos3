# Cosmos3 Piper14 推理资产清单

这个清单只描述推理部署需要的外部模型/缓存资产，供换机器部署时下载和检查。这里不记录当前机器上的绝对路径。

## 放置约定

假设仓库根目录为：

```text
<repo>
```

推荐把资产放到以下相对路径：

```text
<repo>/cosmos_battery/18k
<repo>/cosmos_battery/20k
<repo>/external/cosmos/checkpoints/hf_home
```

启动 server 时通过 `--checkpoint <repo>/cosmos_battery/18k` 或 `--checkpoint <repo>/cosmos_battery/20k` 指定 checkpoint。

## 必需资产

| 资产 | 来源 | 推荐目标位置 | 大小 | 用途 |
|---|---|---|---:|---|
| Piper14 18k checkpoint | 需要团队发布到 HuggingFace/S3/对象存储后填写下载地址 | `cosmos_battery/18k/` | 约 34GB | 18k policy 权重和配置 |
| Piper14 20k checkpoint | 需要团队发布到 HuggingFace/S3/对象存储后填写下载地址 | `cosmos_battery/20k/` | 约 34GB | 20k policy 权重和配置 |
| Wan2.2 VAE | `Wan-AI/Wan2.2-TI2V-5B` | `external/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/...` | 约 3.2GB | Cosmos video tokenizer / VAE |
| Qwen3-VL tokenizer/config | `Qwen/Qwen3-VL-8B-Instruct` | `external/cosmos/checkpoints/hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct/...` | 约 14MB，不含 Qwen safetensors | 文本 tokenizer 和模型配置 |

## Piper14 checkpoint 目录内容

每个 checkpoint 目录至少需要：

```text
checkpoint.json
config.json
model.safetensors.index.json
model-00001-of-00007.safetensors
model-00002-of-00007.safetensors
model-00003-of-00007.safetensors
model-00004-of-00007.safetensors
model-00005-of-00007.safetensors
model-00006-of-00007.safetensors
model-00007-of-00007.safetensors
```

`config.json` 里会引用 Wan2.2 VAE，并声明 `Qwen/Qwen3-VL-8B-Instruct` tokenizer/model name。

## 下载命令示例

安装 HuggingFace CLI 后，可以用下面的命令下载官方依赖：

```bash
HF_HOME=external/cosmos/checkpoints/hf_home \
hf download Wan-AI/Wan2.2-TI2V-5B \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
  --include 'Wan2.2_VAE.pth' \
  --repo-type model
```

```bash
HF_HOME=external/cosmos/checkpoints/hf_home \
hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --exclude '*.safetensors' \
  --exclude '*.bin' \
  --repo-type model
```

如果使用 HuggingFace 默认 cache，而不是手动放到上面的目录，需要保证 Cosmos 运行环境的 `HF_HOME` 指向同一个 cache 根目录。

## 不建议放入 Git 的内容

这些文件是推理需要的大模型权重，但不应作为普通 Git 文件提交：

```text
cosmos_battery/*/model-*.safetensors
external/cosmos/checkpoints/**/Wan2.2_VAE.pth
external/cosmos/checkpoints/**/*.safetensors
external/cosmos/checkpoints/**/*.ckpt
```

原因：

- 单个 checkpoint 约 34GB。
- Wan2.2 VAE 单文件约 3.2GB。
- 普通 GitHub Git push 不适合管理这些大文件。

建议用共享存储、对象存储、HuggingFace model repo 或 Git LFS 管理。

## 最小离线包大小

| 场景 | 估计大小 |
|---|---:|
| 只部署 18k | 约 37.2GB |
| 只部署 20k | 约 37.2GB |
| 同时部署 18k 和 20k | 约 71.2GB |

以上估算包含 Piper14 checkpoint、Wan2.2 VAE 和 Qwen tokenizer/config，不包含 Qwen 原始 safetensors cache。

## 检查方式

下载完成后，至少检查：

```bash
test -f cosmos_battery/18k/config.json
test -f cosmos_battery/18k/model.safetensors.index.json
test -f cosmos_battery/18k/model-00001-of-00007.safetensors
test -f external/cosmos/checkpoints/hf_home/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth
test -f external/cosmos/checkpoints/hf_home/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/tokenizer.json
```

如果只部署 20k，把 `cosmos_battery/18k` 换成 `cosmos_battery/20k`。
