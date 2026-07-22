# Cosmos Piper14 Condition-only VAE 推理優化

## 1. 背景

Cosmos Piper14 policy 每次推理需要一個 33-frame video tensor：第 0 幀是當前觀測，後面 32 幀對應未來 action horizon。部署 wrapper 原本把同一張三相機拼接圖複製 33 次：

```python
video = image.unsqueeze(1).repeat(1, action_rows, 1, 1)
```

位置：`piper_cosmos/deployment/cosmos_piper14_policy.py:152-160`。

舊流程隨後把完整 33 幀送進 Wan2.2 VAE。對 33 個 pixel frames，時間壓縮率為 4，因此 VAE 產生 9 個 latent frames：

```text
33 pixel frames -> 1 + (33 - 1) / 4 = 9 latent frames
```

但 `policy` mode 的 sequence plan 只把 latent frame 0 標為乾淨 vision condition，未來 latent frames 全部是生成目標。定義位於：

- `external/cosmos/cosmos_framework/data/generator/action/transforms.py:289-298`
- `external/cosmos/cosmos_framework/data/generator/action/transforms.py:319-325`

初始化 diffusion noise 時，Cosmos 使用：

```python
noise = condition_mask * x0_token + (1.0 - condition_mask) * pure_noise
```

位置：`external/cosmos/cosmos_framework/model/generator/omni_mot_model.py:1999-2014`。

因此只有 condition mask 為 1 的第 0 幀需要真實 VAE latent。未來 8 個 latent frames 的 `x0_token` 內容會被 condition mask 清除，只需要保留正確的 tensor shape。

## 2. 優化後的資料流

舊流程：

```text
一張 observation image
    -> 複製 33 pixel frames
    -> VAE encode 33 frames
    -> 9 latent frames
    -> frame 0 保留，frame 1-8 被 diffusion noise 覆蓋
```

新流程：

```text
一張 observation image
    -> VAE 只 encode pixel frame 0
    -> 得到 1 個真實 latent frame
    -> 建立完整 9-frame latent tensor
       - latent frame 0：真實 VAE 結果
       - latent frame 1-8：zero placeholders
    -> sampler 按原 condition mask 注入未來 noise
```

這個修改不改變：

- vision latent 的最終 shape；
- action horizon 和 action tensor shape；
- sequence packing 和 temporal position；
- diffusion noise shape 和 seed；
- sampler steps、guidance、shift；
- 最終 action 後處理方式。

## 3. 新增的 condition-only VAE helper

新增檔案：

```text
external/cosmos/cosmos_framework/model/generator/utils/condition_only_vae.py
```

核心函式是 `encode_first_condition_frame()`：

1. 驗證輸入 shape 是 `[B,C,T,H,W]`。
2. 透過 tokenizer 的 `get_latent_num_frames(T)` 計算完整 latent temporal length。
3. 只呼叫一次 `encode(raw_state[:, :, :1])`。
4. 驗證單幀輸入產生一個 latent frame。
5. 配置完整 temporal shape 的 zero tensor。
6. 把真實首幀 latent 複製到 frame 0。

實作位置：`external/cosmos/cosmos_framework/model/generator/utils/condition_only_vae.py:16-65`。

使用 zero placeholder 而不是未初始化記憶體，是為了避免後續 `0 * x0_token` 遇到未定義值或 NaN。

## 4. OmniMoTModel 接入方式

在 `get_data_and_condition()` 的 vision encode 階段加入 fast path：

```text
external/cosmos/cosmos_framework/model/generator/omni_mot_model.py:3264-3297
```

只有同時符合以下條件才會啟用：

- data batch 明確設置 `inference_condition_only_vae=True`；
- 不是 image batch；
- 不是 multi-vision-item batch；
- sequence plan 數量和 vision tensor 數量一致；
- 每個 sequence plan 的 `condition_frame_indexes_vision` 都嚴格等於 `[0]`。

不符合條件時不會改變原推理行為，而是記錄 warning 並回退完整 VAE encode：

```text
OmniMoTModel: condition-only VAE fast path is not eligible for this batch;
falling back to full vision encoding.
```

第一次成功啟用時會記錄：

```text
OmniMoTModel: condition-only VAE fast path enabled;
encoded_pixel_frames=[1] full_pixel_frames=[33] latent_frames=[9]
```

## 5. Piper14 deployment 開關

`CosmosPiper14PolicyConfig` 新增：

```python
condition_only_vae: bool = True
```

位置：`piper_cosmos/deployment/cosmos_piper14_policy.py:26-47`。

Piper backend 在建好 batch 後把這個設定傳給 OmniMoTModel：

```python
batch["inference_condition_only_vae"] = bool(self.config.condition_only_vae)
```

位置：`piper_cosmos/deployment/cosmos_piper14_policy.py:167-168`。

metadata 也會返回 `condition_only_vae`，client 可以確認 server 是否載入新版本：

```text
piper_cosmos/deployment/cosmos_piper14_policy.py:232-243
```

## 6. Server CLI 與回退

新 server 預設啟用 fast path：

```bash
--condition-only-vae
```

如果要做 A/B test 或遇到 action regression，可回退舊版完整 VAE：

```bash
--no-condition-only-vae
```

CLI 定義：`scripts/serve_cosmos_piper14_policy.py:37-42`。

修改程式後必須重啟 policy server；已載入的 Python process 不會自動更新程式。

## 7. 測試

新增測試：

```text
external/cosmos/cosmos_framework/model/generator/utils/condition_only_vae_test.py
```

主要覆蓋：

- 33 pixel frames 只呼叫一次單幀 encoder；
- temporal encoder 輸入長度是 1；
- 輸出 shape 保持為 9 latent frames；
- frame 0 保留 encoder 結果；
- frame 1-8 全部是 zero；
- encoder 如果對一幀產生非 1-frame latent，立即報錯。

Piper backend 測試也檢查 `inference_condition_only_vae=True` 確實進入 data batch：

```text
tests/test_cosmos_piper14_backend_imports.py:241-250
```

本次執行的測試命令：

```bash
cd /home/agilex/World_Action_Model/physical_WM

PYTHONPATH=src/piper-cosmos3/external/cosmos:src/piper-cosmos3 \
/home/agilex/miniconda3/envs/cosmos/bin/python -m pytest -q \
  src/piper-cosmos3/external/cosmos/cosmos_framework/model/generator/utils/condition_only_vae_test.py \
  src/piper-cosmos3/tests/test_cosmos_piper14_backend_imports.py \
  src/piper-cosmos3/tests/test_cosmos_piper14_policy_preprocess.py
```

結果：

```text
9 passed
```

另外對修改檔案執行了 `compileall`，Python 語法檢查通過。

## 8. 正確性依據

Wan2.2 VAE 是 causal encoder。它本身也是先獨立執行單幀 key-frame prime，再處理後續 temporal chunks：

```text
external/cosmos/cosmos_framework/model/generator/tokenizers/wan2pt2_vae_4x16x16.py:802-843
external/cosmos/cosmos_framework/model/generator/tokenizers/wan2pt2_vae_4x16x16.py:920-923
```

因此對相同 frame 0：

```text
encode(video[:, :, :1]) 的唯一 latent
```

應與：

```text
encode(video[:, :, :33]) 的第 0 個 latent
```

使用相同的 causal prime 計算路徑。後續 pixel frames 不會反向影響 frame 0 latent。

這也是 FastWAM action-only inference 採用的策略：只 VAE encode 當前 observation image，不 encode 一段重複的未來 RGB video。

## 9. 如何確認 fast path 正在運作

啟動 server 時明確加入：

```bash
--condition-only-vae
```

client 連線後，metadata 應包含：

```text
'condition_only_vae': True
```

第一次推理後檢查 server log：

```bash
grep -E "condition-only VAE|Starting sampler|Finished sampling" \
  logs/cosmos_server_condition_only_vae.log
```

必須出現：

```text
condition-only VAE fast path enabled
encoded_pixel_frames=[1]
full_pixel_frames=[33]
latent_frames=[9]
```

如果只看到 fallback warning，表示實際 batch 不符合首幀 condition-only 條件，該請求仍走舊版完整 VAE。

## 10. 目前仍存在的成本

這次修改只消除「把 33 個 pixel frames 全部送進 VAE encoder」的成本，尚未消除以下工作：

1. Deployment wrapper 仍會在 CPU 建立 33 張重複的 RGB frame。
2. Transform/normalize 階段仍處理完整 video tensor。
3. 完整 9-frame vision latent tensor 仍需要配置，因為 sequence packing 和 diffusion shape 需要它。
4. Cosmos 仍會生成未來 vision tokens；目前只是不 decode 最終 video。
5. 如果啟用 `COSMOS3_VAE_CPU_OFFLOAD`，每次請求的 VAE 權重 H2D/D2H 搬移仍然存在。
6. Reasoner prefill 是另一條獨立成本，不受本次 VAE 修改影響。

因此不能只根據「33 幀變 1 幀」直接宣稱端到端延遲會縮短 33 倍，必須用相同 observation、seed、steps、guidance 和 GPU 配置做 A/B benchmark。

## 11. 建議的後續優化

完成本次首幀 VAE fast path 後，下一階段可依序處理：

1. 在 transform 完成 sequence plan 後，只保留一張 RGB condition frame，另外傳遞 logical pixel-frame count，消除 33-frame CPU/GPU normalize 和傳輸。
2. 若顯存允許，評估 VAE GPU resident，消除每次 VAE 權重搬移。
3. 對固定 instruction 實作跨請求 Reasoner UND K/V cache。
4. 研究 FastWAM 式 vision prefill + action-only denoise，避免每步更新未來 vision branch。
5. 移除本地 backend 未使用的 PNG/base64 建構與 RPC `.tolist()` 轉換。

## 12. 本次修改檔案

```text
external/cosmos/cosmos_framework/model/generator/utils/condition_only_vae.py
external/cosmos/cosmos_framework/model/generator/utils/condition_only_vae_test.py
external/cosmos/cosmos_framework/model/generator/omni_mot_model.py
piper_cosmos/deployment/cosmos_piper14_policy.py
scripts/serve_cosmos_piper14_policy.py
tests/test_cosmos_piper14_backend_imports.py
```

