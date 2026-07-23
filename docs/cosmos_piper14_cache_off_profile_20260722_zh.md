# Cosmos Piper14 Cache-off 推理 Profile 報告

## 1. 測試目的

測量 condition-only VAE 開啟、instruction cache 關閉時，Cosmos Piper14 單次推理的端到端延遲、模型內部階段耗時及 CUDA 顯存使用。

本報告回答三個問題：

1. 單次端到端推理需要多久。
2. 資料處理、Reasoner prefill 和四步 diffusion denoise 各佔多少時間。
3. 24 GB GPU 是否存在顯存洩漏，以及還有多少峰值餘量。

## 2. 測試配置

服務器核心配置：

```text
checkpoint: cosmos_battery/20k
resolution: 480
camera input: 3 × 480 × 640 RGB
action horizon: 32
raw action dim: 14
precision: bfloat16
sampler: UniPC
num_steps: 4
guidance: 3.0
shift: 5.0
seed: 0
condition_only_vae: true
instruction_cache: false
reasoner_offload: true
vae_cpu_offload: true
timing: true
cuda_memory: true
```

Client 使用 `episode_126.hdf5`，連續發送 12 個純推理 request，沒有連接或控制機械臂。Request 1 包含 allocator-history dump，request 1–2 視為 warm-up；穩態統計使用 request 3–12，共 10 次。

測試輸出：

```text
logs/instruction_cache_ab/server_cache_off_profile.log
logs/instruction_cache_ab/cache_off_cuda_history.pickle
output_actions/profile_cache_off_no_action.npz
```

## 3. 端到端結果

### 3.1 Client E2E

穩態 request 3–12：

| 指標 | 延遲 |
|---|---:|
| 平均 | 6475.4 ms |
| 中位數 | 6467.6 ms |
| 最小 | 6396.0 ms |
| 最大 | 6535.3 ms |
| 標準差 | 40.1 ms |

標準差約為平均值的 0.62%，推理延遲穩定。

### 3.2 Server policy

| 階段 | 平均耗時 | 佔 server policy total |
|---|---:|---:|
| `policy.total` | 6472.5 ms | 100.0% |
| `backend.total` | 6392.4 ms | 98.8% |
| `model.generate.total` | 6277.5 ms | 97.0% |
| `model.prepare.total` | 524.7 ms | 8.1% |
| `model.reasoner.prefill[2]` | 2436.0 ms | 37.6% |
| `model.denoise.velocity[8]` | 3298.0 ms | 51.0% |

Client E2E 與 server `policy.total` 平均只相差約 2.9 ms，因此本機 RPC 傳輸不是瓶頸。

## 4. 非重疊延遲拆分

將嵌套 timer 改寫成非重疊階段後：

```text
Policy request 準備                     約   80.1 ms
Backend/model 外處理                   約  114.9 ms
Model data preparation                 約  524.7 ms
兩次 CFG Reasoner prefill              約 2436.0 ms
八次 CFG velocity forward              約 3298.0 ms
Model 其餘 sampler/調度開銷             約   18.7 ms
────────────────────────────────────────────────
Server policy total                    約 6472.5 ms
```

主要瓶頸不是 RPC 或資料處理，而是 Reasoner prefill 和 CFG denoise forward；兩者合計約佔 88.6%。

## 5. 資料處理結果

穩態平均：

| 階段 | 平均耗時 |
|---|---:|
| `request.concat_view` | 2.8 ms |
| `request.png_base64` | 76.7 ms |
| `backend.video_repeat` | 2.6 ms |
| `backend.transform` | 106.4 ms |
| `model.prepare.vision_action` | 515.6 ms |
| `model.prepare.text` | 0.075 ms |
| `model.prepare.pack[11]` | 約 9 ms |

包含 VAE、transform 和 request 構建的完整資料準備約為 0.72 秒。Condition-only VAE 已正確啟用：

```text
encoded_pixel_frames=[1]
full_pixel_frames=[33]
latent_frames=[9]
```

這證明原來約 11 秒的完整 33-frame VAE/data 路徑已不再是主要瓶頸。

目前仍有兩項較小的冗餘：

1. 本地 backend 已直接傳遞 `concat_view`，但 request 仍建立 PNG/base64，約 77 ms。
2. Wrapper 仍建立 33-frame repeated tensor 並執行完整 transform，約消耗 0.1 秒及額外臨時顯存。

即使完全消除這兩項，收益也只有約 0.1–0.2 秒。

## 6. Reasoner prefill 與四步 denoise

本次 `guidance=3.0`，因此啟用 classifier-free guidance。模型為 conditional 和 unconditional prompt 分別建立 Reasoner memory：

```text
conditional Reasoner prefill      1 次
unconditional Reasoner prefill    1 次
─────────────────────────────────────
合計                               2 次
```

兩次 prefill 平均累計 2436.0 ms，即每個分支約 1218.0 ms。

UniPC 有四個 diffusion steps。每一步都需要 conditional 和 unconditional velocity：

```text
4 diffusion steps × 2 CFG branches = 8 velocity forwards
```

八次 velocity 累計 3298.0 ms：

```text
每個 velocity forward：約 412.3 ms
每個完整 CFG diffusion step：約 824.5 ms
純四步 velocity 計算：約 3.30 秒
```

TQDM 顯示的約 5.7 秒 sampling 包含在第一次 sampler callback 中執行的兩次 Reasoner prefill。因此不能把 TQDM 的完整 5.7 秒全部稱為「四步去噪」：

```text
Reasoner prefill：約 2.44 秒
四步 CFG velocity：約 3.30 秒
其他 sampler 開銷：約 0.02 秒
```

## 7. Instruction cache 狀態

本輪配置為 `--no-instruction-cache`：

```text
request 數：12
每 request prefill：2
UND-only prefill 總數：24
instruction memory cache hit：0
```

若假設兩次 prefill 可以被完整且正確地消除，算術上的 E2E 下界為：

```text
6472.5 ms - 2436.0 ms = 4036.5 ms
```

這只是理論上限。是否能復用 cache，必須用相同輸入做 cache-on A/B，並驗證輸出 action 的數值一致性和新觀測是否仍被正確處理。

## 8. CUDA 顯存結果

穩態 allocator 數據：

| 指標 | 顯存 |
|---|---:|
| request baseline allocated | 15701 MiB |
| request peak allocated | 17697 MiB |
| request final allocated | 15701 MiB |
| allocator reserved | 約 17740 MiB |
| GPU total | 24078 MiB |
| peak headroom | 約 6381 MiB / 6.23 GiB |
| peak percentage | 73.5% |

Request 2–12 結束後 allocated 都回到 15701 MiB，沒有逐 request 增長，未觀察到 CUDA allocator 顯存洩漏。

Vision/VAE preparation 產生約 1996 MiB 的瞬時峰值。Allocator history 中最大的單次 allocation 包括：

```text
Wan2.2 VAE forward：265.1 MiB
video normalization：3 × 154.7 MiB
Reasoner layer staging：單次最高 96 MiB
```

Allocator history 共記錄 74,824 個事件，低於 200,000 上限，沒有因上限被截斷：

```text
alloc：24927
free_requested：24922
free_completed：24922
segment_map：53
```

大量累計 allocation traffic 來自 Qwen/MoT forward。Reasoner layer staging 的累計 allocation traffic 約為 25.9 GiB；這是整次推理期間反覆配置的累計流量，不是同時佔用量。

Snapshot 時：

```text
reserved：17738 MiB
allocated/active：15858 MiB
allocator 保留但非 active：約 1880 MiB
```

## 9. 第一個 request

第一個 request 的 client E2E 為 8.91 秒，不能納入穩態平均。它包含：

1. 首次模型 warm-up/compile。
2. CUDA allocator history 記錄開銷。
3. 約 32.7 MB `.pickle` snapshot 的序列化和寫盤。

Snapshot dump 沒有獨立 timing stage，因此其寫盤時間包含在 `backend.total` 的未分類部分。

## 10. 優化優先級

1. 對 instruction cache 做 cache-on A/B，直接測量 2.44 秒 prefill 是否能安全消除。
2. 測試 `guidance=1.0`，它可以取消 unconditional 分支，但必須評估 action/策略品質。
3. 研究 action-only 或減少未來 vision-token denoise，降低八次 Qwen/MoT velocity forward 的計算量。
4. 在目前約 6.23 GiB 峰值餘量下 A/B 測試 VAE GPU resident，確認能否降低 0.52 秒 vision/action preparation。
5. 最後再移除 PNG/base64、33-frame CPU repeat/transform 等約 0.1–0.2 秒的小型開銷。

