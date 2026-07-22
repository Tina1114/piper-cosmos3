# Cosmos Piper14 固定 Instruction Cache 優化

## 目標

同一個真機任務的 prompt 在連續 policy chunk 之間不變。第 1 個 chunk 正常完成文字處理和 Reasoner understanding prefill；第 2 個 chunk 開始重用結果，圖片、機器人 state、video/action latent 和 diffusion generation 仍然逐次重新計算。

## 快取內容

1. Qwen chat template/tokenize 產生的 token IDs。
2. Reasoner 每一層 understanding K/V。

第二項是主要優化。Cosmos 不是 FastWAM 的獨立 T5 encoder 架構，所以只保存一個 text embedding 不足以跳過 Reasoner；需要保存固定 instruction 經過 understanding tower 後的逐層 K/V。

## 預設安全模式

Policy server 預設開啟 `--instruction-cache`，但不指定 `--instruction-cache-dir`：

- 每次 server 啟動後，第 1 個 chunk 一定重新計算並建立 process-local cache。
- 第 2 個 chunk 起才重用。
- server 重啟後 cache 消失，下一次第 1 個 chunk 重新建立。
- prompt token、checkpoint/config namespace、dtype、Reasoner 層數任一項不同都不會命中。
- 圖片和 state 不在這個 cache 中，仍會隨每個 chunk 更新。

啟動命令不需要增加參數：

```bash
bash scripts/start_cosmos_piper14_20k_server.sh
```

如需完全停用，以原始路徑做 A/B 驗證：

```bash
python scripts/serve_cosmos_piper14_policy.py \
  --checkpoint /home/agilex/World_Action_Model/physical_WM/checkpoints/cosmos_battery/20k \
  --config-file configs/cosmos_piper14_20k_local_config.json \
  --no-instruction-cache
```

## 可選磁碟模式

只有明確傳入以下參數才會跨 server 重啟保存 cache：

```bash
--instruction-cache-dir /home/agilex/World_Action_Model/physical_WM/data/cosmos_instruction_cache
```

目前以 process-local 模式作為真機預設，先確保第 1 個 chunk 完整計算、第 2 個 chunk 才命中。

## 預期日誌

第 1 個 chunk 建立完成後，後續 chunk 會看到類似：

```text
OmniMoTModel: instruction Reasoner K/V memory cache hit (...)
```

使用磁碟模式時還會看到 text-token/KV cache 的 save 或 disk cache hit 日誌。

## 真機前驗證

先保持 `--no-execute-actions`，固定相同圖片/state/seed，各跑 cache 關閉與開啟兩次，核對：

1. action shape 都是 `[32, 14]`。
2. action 中沒有 NaN/Inf。
3. 開啟 cache 後，第 2 個 chunk 出現 memory cache hit。
4. 兩條路徑輸出的 action 在既定容差內一致，再允許真機執行。
