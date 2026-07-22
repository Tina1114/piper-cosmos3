# Cosmos Piper14 VAE GPU Resident 單次無動作測試（2026-07-22）

## 1. 測試目的

在不連接、不控制 Piper 真機的情況下，使用固定 HDF5 輸入對重新啟動的 Cosmos Piper14 服務執行一次推理，確認服務可用、輸出合法，並記錄 timing 配置與端到端延遲。

## 2. Server metadata

測試前從 `127.0.0.1:8766` 讀得：

| 配置 | 值 |
|---|---:|
| checkpoint | `cosmos_battery/20k` |
| action horizon | 32 |
| resolution | 480 |
| denoise steps | 4 |
| CFG guidance | 3.0 |
| shift | 5.0 |
| condition-only VAE | true |
| instruction cache | true |
| segmented timing | true |
| CUDA memory timing | true |

服務進程參數中 `--cuda-memory-history` 後面的值被解析成了字串 `" 2"`，不是預期的 `.pickle` 文件路徑。原因通常是 shell 命令把 `2>&1` 的 `2` 誤當成了該參數的值。下次應顯式寫成：

```bash
--cuda-memory-history logs/instruction_cache_ab/vae_gpu_resident_cuda_history.pickle \
2>&1 | tee logs/instruction_cache_ab/server_vae_gpu_resident_profile.log
```

## 3. 無動作測試輸入

```text
episode: data/battery_assemble/perfect/episode_126.hdf5
frame: 0
prompt: Assemble the mouse's battery.
execute actions: false
```

輸出保存在：

```text
output_actions/profile_vae_gpu_resident_single_no_action.npz
```

## 4. 本次結果

| 指標 | 時間 |
|---|---:|
| metadata RPC | 44.110 ms |
| infer RPC send | 2.527 ms |
| infer server/recv wait | 8731.183 ms |
| client infer E2E | **8733.713 ms** |

輸出 action shape 為 `[32, 14]`，未下發到機械臂。數值檢查正常：

```text
pred_vs_gt_mae = 0.027423
pred_first_vs_qpos_linf = 0.016183
pred_internal_linf = 0.058844
```

這是新服務的第一個 inference request，包含 instruction-cache miss、首次模型 warm-up/compile，以及 CUDA allocator history 初始化/錯誤路徑相關開銷，因此不能視為第二個 chunk 開始的穩態延遲。

## 5. Server 分段時間的可觀測性限制

本輪 server 確實啟用了 `--timing --cuda-memory`，但 `[cosmos-piper14-timing]` 與 `[cosmos-piper14-cuda-memory]` 只輸出到啟動服務的容器終端，沒有 `tee` 到共享工作區。當前 host 帳號也沒有讀取 Docker daemon logs 的權限，因此無法從 client 輸出可靠還原 `prepare / prefill / denoise / transform` 的實測分段數字。

本輪只把可驗證的 E2E 數字寫入文檔，不使用舊 profile 的分段結果代替。若要保存下一次分段結果，server 啟動命令末尾必須包含：

```bash
2>&1 | tee logs/instruction_cache_ab/server_vae_gpu_resident_profile.log
```

完成一次推理後可提取：

```bash
rg 'cosmos-piper14-(timing|cuda-memory)' \
  logs/instruction_cache_ab/server_vae_gpu_resident_profile.log
```

## 6. 結論

新服務已通過單次無動作 smoke test，RPC、三路圖像輸入與 `[32,14]` action 輸出均正常。第一個 request E2E 為 8.734 秒；由於它是 cache miss/warm-up request，真機連續運行時應重點觀察第二個 request 之後的穩態延遲。

## 7. 同一服務進程的多 chunk 測試

單次 smoke test 後沒有重啟服務，因此上一節的 8.734 秒正好是該進程的 Chunk 1 冷啟動。隨後使用相同 prompt 及同一 HDF5 episode 連續執行 12 個無動作 chunk，作為 instruction-cache hit 的穩態樣本。

輸出文件：

```text
output_actions/profile_vae_gpu_resident_multichunk_no_action.npz
```

完整 E2E 延遲：

```text
冷啟動 Chunk 1：8733.758 ms

後續 cache-hit chunks：
3621.475, 3599.931, 3609.077, 3623.192,
3632.116, 3644.528, 3691.250, 3662.881,
3677.191, 3713.288, 3662.503, 3668.979 ms
```

穩態統計：

| 指標 | 時間 |
|---|---:|
| 樣本數 | 12 |
| mean | **3650.534 ms** |
| P50 | 3653.515 ms |
| P95 | 3701.167 ms |
| standard deviation | 33.153 ms |
| min | 3599.931 ms |
| max | 3713.288 ms |

冷啟動比穩態平均慢 5083.224 ms，約為穩態的 2.39 倍。與舊的 VAE CPU offload、instruction-cache off 穩態基線 6475.400 ms 相比，目前組合配置快 2824.866 ms，即 1.77 倍；但因為 VAE placement 與 instruction cache 同時改變，不能把全部差值歸因於 VAE 常駐。

新舊測試的 12 組 `[32,14]` predicted actions 逐元素完全一致：

```text
max_abs_diff = 0.0
```

本節的正確解讀是：同一 server process 的第一個 cold/cache-miss chunk 為 8.734 秒；從第二個服務請求開始，cache-hit E2E 穩定在約 3.65 秒。由於 server stdout 仍未寫入共享日志，本節依然沒有用推測值填寫 VAE、prefill、denoise 的內部分段時間。
