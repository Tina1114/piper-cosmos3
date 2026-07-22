# Cosmos Piper14 服務器與客戶端啟動配置指南

本文以目前倉庫中的 Cosmos Piper14 20k policy 為準，說明三部分：環境配置、服務器端配置、客戶端配置。服務器與客戶端可以位於同一台機器，也可以位於可信局域網內的不同機器。

預設接口如下：

```text
RPC: Python multiprocessing.connection
host: 服務器 IP
port: 8766
authkey: cosmos-piper14
輸入: 3 路 RGB + 14 維雙臂 state + prompt
輸出: [32, 14] absolute joint-position action chunk
```

> `authkey` 只用於 RPC 身份校驗，不提供傳輸加密。不要把端口直接暴露到不可信網絡。

## 1. 環境配置

### 1.1 目錄與 Python 環境

倉庫已提供統一環境腳本：

```bash
cd /home/agilex/World_Action_Model/physical_WM/src/piper-cosmos3
source scripts/env_cosmos_piper14_20k.sh
```

腳本會設置以下變量：

| 變量 | 功能 |
|---|---|
| `REPO_ROOT` | `piper-cosmos3` 倉庫根目錄 |
| `PHYSICAL_WM_ROOT` | `physical_WM` 根目錄 |
| `CHECKPOINT_DIR` | Cosmos Piper14 20k inference checkpoint |
| `CONFIG_FILE` | Cosmos 模型 JSON 配置 |
| `HF_HOME` | Hugging Face 本地模型緩存 |
| `COSMOS3_QWEN_SNAPSHOT` | Qwen3-VL-8B-Instruct 本地 snapshot |
| `COSMOS3_WAN_VAE_PATH` | Wan2.2 VAE 權重文件 |
| `COSMOS3_FRAMEWORK_ROOT` | Cosmos framework 源碼路徑 |
| `PYTHONPATH` | 加入本倉庫和 Cosmos framework |
| `COSMOS_PIPER14_PYTHON` | 實際啟動服務器的 Python 解釋器 |
| `HF_HUB_OFFLINE=1` | 禁止 Hugging Face 在線下載 |
| `TRANSFORMERS_OFFLINE=1` | 強制 Transformers 使用本地資產 |
| `XDG_CACHE_HOME` 等 | 編譯及 runtime cache 位置 |

執行完整檢查：

```bash
bash scripts/check_cosmos_piper14_20k_env.sh
```

另行確認版本與 GPU：

```bash
git rev-parse HEAD

"${COSMOS_PIPER14_PYTHON}" -c \
'import torch; print("torch", torch.__version__); print("cuda", torch.version.cuda); print("available", torch.cuda.is_available()); print("gpu", torch.cuda.get_device_name(0))'

nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv
```

### 1.2 GPU、offload 與 cache 環境變量

推薦先明確清除所有可能從 shell 繼承的舊設定，再設置本次實驗需要的值：

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

unset COSMOS3_LAYER_OFFLOAD
unset COSMOS3_REASONER_KVCACHE
unset COSMOS3_REASONER_KVCACHE_DEBUG
unset COSMOS3_REASONER_EMPTY_CACHE
unset COSMOS3_REASONER_OFFLOAD
unset COSMOS3_VAE_CPU_OFFLOAD
unset COSMOS3_VAE_GPU_RESIDENT
```

各參數含義：

| 變量 | 建議值 | 功能與注意事項 |
|---|---:|---|
| `CUDA_VISIBLE_DEVICES` | `0` | 固定服務器使用哪張物理 GPU |
| `COSMOS3_REASONER_OFFLOAD` | `1` | 單卡顯存模式：Reasoner 層按需搬到 GPU，prefill 後保留 generator；目前只支持 single-rank inference |
| `COSMOS3_REASONER_KVCACHE` | 通常不手動設 | 啟用 diffusion Reasoner K/V memory path；`--instruction-cache` 會在模型建立前自動設置 |
| `COSMOS3_REASONER_KVCACHE_DEBUG` | 默認關 | 額外驗證 K/V cache，僅排錯使用，可能增加開銷 |
| `COSMOS3_REASONER_EMPTY_CACHE` | 默認關 | Reasoner 階段的額外 CUDA cache 清理策略；只應在 OOM 排錯時 A/B 驗證 |
| `COSMOS3_VAE_CPU_OFFLOAD` | `1`（24 GB 單卡常用） | VAE 平時放 CPU，encode 時搬到 GPU；節省顯存但增加 H2D/D2H 時間 |
| `COSMOS3_VAE_GPU_RESIDENT` | 顯存足夠時 `1` | VAE 常駐 GPU；它會覆蓋 `COSMOS3_VAE_CPU_OFFLOAD` |
| `COSMOS3_LAYER_OFFLOAD` | 默認關 | 更通用的逐層 offload；不要在未驗證時與 Reasoner offload 混用 |
| `COSMOS_PIPER14_CLIENT_TIMING` | client 可設 `1` | 打印 RPC send、recv wait 和 total latency，不影響 server |

目前 24 GB 單卡基線：

```bash
export CUDA_VISIBLE_DEVICES=0
export COSMOS3_REASONER_OFFLOAD=1
export COSMOS3_VAE_CPU_OFFLOAD=1
unset COSMOS3_VAE_GPU_RESIDENT
unset COSMOS3_LAYER_OFFLOAD
```

若顯存足夠並要測試 VAE 常駐：

```bash
unset COSMOS3_VAE_CPU_OFFLOAD
export COSMOS3_VAE_GPU_RESIDENT=1
```

兩種 VAE placement 必須分開測試，不可把結果直接混合比較。

### 1.3 客戶端環境

HDF5 回放 client 需要 Python、NumPy、h5py、YAML/OmegaConf 及本倉庫 `PYTHONPATH`。真機 client 還需要 ROS Noetic、`rospy`、`cv_bridge`、三路 RGB topic、`pyAgxArm` 和兩個 CAN interface。

真機 client shell 通常先執行：

```bash
source /opt/ros/noetic/setup.bash
cd /home/agilex/World_Action_Model/physical_WM/src/piper-cosmos3
export PYTHONPATH="${PWD}:${PYTHONPATH}"
```

如果使用 conda，必須在同一 shell 中激活包含 ROS bridge、pyAgxArm 和本倉庫依賴的環境。不要讓 server 與 client 意外使用不同的 Python。

網絡必須允許 client 連接 server 的 TCP 端口：

```bash
nc -vz <SERVER_IP> 8766
```

## 2. 服務器端配置

### 2.1 CLI 參數與功能

服務器入口：

```text
scripts/serve_cosmos_piper14_policy.py
```

| 參數 | 默認值 | 功能 |
|---|---:|---|
| `--checkpoint` | 腳本內默認路徑 | inference checkpoint 目錄 |
| `--config-file` | `None` | Cosmos 模型 JSON；部署 20k checkpoint 時應明確傳入 |
| `--host` | `127.0.0.1` | 綁定地址；跨機連接使用 `0.0.0.0` |
| `--port` | `8766` | RPC TCP 端口，client 必須一致 |
| `--authkey` | `cosmos-piper14` | RPC 共享密鑰，client 必須一致 |
| `--prompt` | battery prompt | server 默認 prompt；request 內 prompt 可覆蓋 |
| `--action-horizon` | `32` | 每次返回的未來 action 數量 |
| `--max-action-dim` | `64` | 模型內部 action padding 維度；Piper 原始輸出仍為 14 |
| `--num-steps` | `4` | UniPC 去噪步數；越少越快，但可能降低策略品質 |
| `--guidance` | `3.0` | CFG guidance；不等於 1 時通常計算 conditional 和 unconditional 分支 |
| `--shift` | `5.0` | rectified-flow/UniPC inference shift |
| `--fps` | `30` | conditioning FPS，影響時間位置編碼 |
| `--seed` | `0` | diffusion 初始噪聲 seed；固定可提高 A/B 可比性 |
| `--camera-height` | `480` | 每路輸入相機期望高度 |
| `--camera-width` | `640` | 每路輸入相機期望寬度 |
| `--resolution` | `480` | Cosmos transform/生成 resolution 類別 |
| `--condition-only-vae` | 開 | 只 VAE encode 第 0 個 condition frame，再補完整 latent shape |
| `--no-condition-only-vae` | — | 回退完整 33-frame VAE encode，用於正確性與性能 A/B |
| `--instruction-cache` | 開 | 第 2 個相同 prompt chunk 起復用 text UND/Reasoner K/V |
| `--no-instruction-cache` | — | 每個 chunk 重新建立 instruction K/V |
| `--instruction-cache-dir` | `None` | 可選磁碟 cache；正常測試建議不設，使用 process-local cache |
| `--instruction-cache-max-entries` | `4` | process-local instruction cache 最大 prompt 數 |
| `--mock-backend` | 關 | 不載入真模型的 RPC/API 測試 backend，不能用於性能或策略品質測試 |
| `--timing` | 關 | 每個 inference 以 CUDA synchronize 分段計時，打印 request、資料準備、Reasoner、denoise 和輸出階段 |
| `--cuda-memory` | 關 | 在分段邊界採樣 CUDA allocator/driver 顯存並打印 request peak；同時會啟用分段 timer |
| `--cuda-memory-history <path>` | `None` | 啟動 PyTorch CUDA allocator history，完成第一個 inference 後 dump `.pickle` snapshot，之後自動停止記錄 |
| `--cuda-memory-history-max-entries` | `200000` | allocator history 最多保留的事件數；越大越完整，也會消耗更多 host memory |

### 2.2 推薦服務器啟動命令

建立日志目錄：

```bash
cd /home/agilex/World_Action_Model/physical_WM/src/piper-cosmos3
source scripts/env_cosmos_piper14_20k.sh

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export COSMOS3_REASONER_OFFLOAD=1
export COSMOS3_VAE_GPU_RESIDENT=1
unset COSMOS3_VAE_CPU_OFFLOAD=1
unset COSMOS3_LAYER_OFFLOAD

mkdir -p logs/deployment
```

正常服務器（condition-only VAE 開、instruction cache 開）：

```bash
"${COSMOS_PIPER14_PYTHON}" scripts/serve_cosmos_piper14_policy.py \
  --checkpoint "${CHECKPOINT_DIR}" \
  --config-file "${CONFIG_FILE}" \
  --host 0.0.0.0 \
  --port 8766 \
  --authkey cosmos-piper14 \
  --condition-only-vae \
  --instruction-cache \
  --instruction-cache-max-entries 4 \
  --action-horizon 32 \
  --num-steps 4 \
  --guidance 3.0 \
  --shift 5.0 \
  --fps 30 \
  --seed 0 \
  --resolution 480 \
  --camera-height 480 \
  --camera-width 640 \
  2>&1 | tee logs/deployment/cosmos_piper14_server.log
```

看到以下輸出才表示可以啟動 client：

```text
[cosmos-piper14-policy-server] Listening on 0.0.0.0:8766
```

快速啟動腳本等價於使用大多數默認參數：

```bash
bash scripts/start_cosmos_piper14_20k_server.sh
```

正式 benchmark 建議使用上面的完整命令，避免依賴隱式默認值。

### 2.3 分段計時與 CUDA 顯存 profile

三個 profiler 開關的用途不同：

```text
--timing             每個 request 都打印分段 wall time
--cuda-memory        每個 request 都打印各階段 CUDA memory snapshot
--cuda-memory-history 只記錄並 dump 第一個完成的 inference
```

`--timing` 和 `--cuda-memory` 會在階段邊界執行 `torch.cuda.synchronize()`，因此適合找瓶頸，不代表 profiler 關閉時的最低延遲。`--cuda-memory-history` 本身也有記錄開銷，不應拿該次 request 當正式 latency 結果。

分段 timer 目前包含：

| 日志 stage | 內容 |
|---|---|
| `observation.coerce` | 驗證三路影像、state 和 prompt |
| `request.concat_view` | 拼接三路相機影像 |
| `request.png_base64` | 建立兼容 RPC request 的 PNG/base64 欄位 |
| `backend.video_repeat` | 建立 33-frame logical video tensor |
| `backend.transform` | resize/normalize/action transform |
| `backend.batch_build` | 建立 Cosmos data batch |
| `model.prepare.vision_action` | VAE encode 及 vision/action condition 準備 |
| `model.prepare.text` | prompt tokenize/cache lookup |
| `model.prepare.pack` | multimodal sequence packing；可能被多次調用 |
| `model.prepare.total` | 完整 inference data 準備 |
| `model.reasoner.memory` | 建立 Reasoner memory state |
| `model.reasoner.restore` | offload 模式下準備 Reasoner prefill |
| `model.reasoner.prefill` | UND/Reasoner prefill；CFG 通常有兩次 |
| `model.reasoner.offload` | Reasoner offload/finalize；可能被多次調用 |
| `model.denoise.velocity` | sampler 的 velocity forward 累計時間與調用次數 |
| `model.generate.total` | 完整 model generation |
| `backend.action_output` | action slice、GPU→CPU 和 list conversion |
| `policy.total` | server policy 從收到 observation 到完成 action 的總時間 |

只做多 request 分段計時和顯存 peak，推薦使用：

```bash
--timing \
--cuda-memory
```

同時抓取第一個 inference 的 allocator history：

```bash
--timing \
--cuda-memory \
--cuda-memory-history logs/instruction_cache_ab/cache_off_cuda_history.pickle \
--cuda-memory-history-max-entries 200000
```

完整的 cache-off 診斷啟動命令：

```bash
"${COSMOS_PIPER14_PYTHON}" scripts/serve_cosmos_piper14_policy.py \
  --checkpoint "${CHECKPOINT_DIR}" \
  --config-file "${CONFIG_FILE}" \
  --host 0.0.0.0 \
  --port 8766 \
  --authkey cosmos-piper14 \
  --condition-only-vae \
  --no-instruction-cache \
  --num-steps 4 \
  --guidance 3.0 \
  --shift 5.0 \
  --seed 0 \
  --resolution 480 \
  --camera-height 480 \
  --camera-width 640 \
  --action-horizon 32 \
  --timing \
  --cuda-memory \
  --cuda-memory-history logs/instruction_cache_ab/cache_off_cuda_history.pickle \
  --cuda-memory-history-max-entries 200000 \
  2>&1 | tee logs/instruction_cache_ab/server_cache_off_profile.log
```

完成第一個 inference 後應看到：

```text
[cosmos-piper14-timing] request=1 ...
[cosmos-piper14-cuda-memory] request=1 ...
[cosmos-piper14-cuda-memory-stage] request=1 stage=... ...
[cosmos-piper14-cuda-memory] snapshot=logs/instruction_cache_ab/cache_off_cuda_history.pickle
```

重啟後先用 metadata 確認 profiler 確實載入：

```text
timing: True
cuda_memory: True
cuda_memory_history: logs/instruction_cache_ab/cache_off_cuda_history.pickle
```

如果目標是穩態 latency，建議另啟一輪只帶 `--timing` 的 server，先 warm-up 兩次，再統計後續 requests。不要把 allocator-history request、compile warm-up request 和穩態 request 混成同一平均值。

### 2.4 A/B 測試命令差異

測試 condition-only VAE 時，固定 instruction cache 關閉，只切換：

```text
--no-condition-only-vae
--condition-only-vae
```

測試 instruction cache 時，固定 condition-only VAE 開啟，只切換：

```text
--no-instruction-cache
--instruction-cache
```

每組都必須重啟 server。不要在同一個 server process 中混合冷啟動、compile warm-up、cache miss 和 cache hit 結果。

### 2.5 服務器驗收

client 連接後檢查 metadata：

```text
raw_action_dim: 14
action_horizon: 32
image_keys: cam_high, cam_left_wrist, cam_right_wrist
condition_only_vae: True（正常優化配置）
```

condition-only VAE 第一次推理應出現：

```text
condition-only VAE fast path enabled
encoded_pixel_frames=[1]
full_pixel_frames=[33]
latent_frames=[9]
```

instruction cache 開啟時，第 2 個相同 prompt chunk 起才應出現 memory cache hit。這個 cache 不會跳過新影像、新 state、VAE 或四步 denoising。

## 3. 客戶端配置

客戶端有三種推薦啟動方式：RPC 冒煙測試、HDF5 性能/正確性測試、ROS 真機 RTC。

### 3.1 RPC 輸入輸出規格

每次 `infer` observation 必須包含：

| 欄位 | 格式 | 說明 |
|---|---|---|
| `images.cam_high` | `uint8[480,640,3]` | 頭部 RGB |
| `images.cam_left_wrist` | `uint8[480,640,3]` | 左腕 RGB |
| `images.cam_right_wrist` | `uint8[480,640,3]` | 右腕 RGB |
| `state` | `float32[14]` | 左臂 6 joint + gripper，右臂 6 joint + gripper |
| `prompt` | `str` | 任務指令；cache 測試時必須固定 |

返回：

```text
float32[32,14]
```

它表示 32 個未來 absolute joint-position command，不是 delta action。

### 3.2 最小 RPC/metadata 冒煙測試

在 client 機器執行：

```bash
cd /home/agilex/World_Action_Model/physical_WM/src/piper-cosmos3
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python3 -c 'from piper_cosmos.deployment.cosmos_piper14_remote_client import CosmosPiper14RemotePolicyClient as C; c=C(host="<SERVER_IP>", port=8766, authkey="cosmos-piper14"); print(c.metadata()); c.close()'
```

這只檢查連接和 metadata，不會執行模型推理。

### 3.3 HDF5 回放：推薦的安全測試 client

入口：

```text
scripts/replay_hdf5_frames_to_real_piper14.py
```

主要參數：

| 參數 | 默認值 | 功能 |
|---|---:|---|
| `--episode` | 必填 | 輸入 HDF5 episode |
| `--data-config` | battery dataset config | HDF5 image/qpos/action key mapping |
| `--config` | real deployment YAML | server、robot 及安全配置 |
| `--host` | YAML `policy_server.host` | 覆蓋服務器 IP |
| `--port` | `8766` | RPC port |
| `--authkey` | `cosmos-piper14` | RPC authkey |
| `--prompt` | dataset/default | 覆蓋任務 prompt |
| `--stride` | `32` | 每隔多少 dataset frames 發起一次推理 |
| `--max-anchors` | 不限制 | 最多推理多少個 anchor，適合 smoke test |
| `--output` | `output_actions/hdf5_replay_predictions.npz` | 保存 action、GT 和 E2E latency |
| `--execute-actions` | 關 | 打開後會真的控制雙臂；性能測試禁止使用 |
| `--move-to-dataset-start` | 關 | 真機執行前移至 dataset 起始 qpos |
| `--reset-to-dataset-qpos-each-anchor` | 關 | 每個 anchor 前回到對應 dataset qpos |
| `--yes` | 關 | 跳過真機危險操作確認；正常不要使用 |

安全 smoke test：

```bash
export COSMOS_PIPER14_CLIENT_TIMING=1

python3 scripts/replay_hdf5_frames_to_real_piper14.py \
  --episode data/battery_assemble/perfect/episode_126.hdf5 \
  --data-config configs/dataset_configs/battery_assemble_hdf5.yaml \
  --config configs/real_deploy_cosmos_battery_motion.yaml \
  --host <SERVER_IP> \
  --port 8766 \
  --authkey cosmos-piper14 \
  --stride 32 \
  --max-anchors 3 \
  --output output_actions/cosmos_smoke_test.npz
```

性能測試可把 `--max-anchors` 提高到至少 22，丟棄前兩次 warm-up，再統計 `inference_latency_s`。

### 3.4 ROS／真機 RTC YAML 配置

真機入口：

```text
scripts/run_real_cosmos_piper14_runtime.py
```

請使用：

```text
configs/real_deploy_cosmos_battery_motion.yaml
```

Cosmos client 實際讀取以下 YAML 欄位。

#### `policy_server`

| 欄位 | 功能 |
|---|---|
| `host` | Cosmos server IP；同機為 `127.0.0.1` |
| `port` | RPC port，必須與 server 一致 |
| `authkey` | RPC 共享密鑰，必須與 server 一致 |

#### `fastwam`

此節名稱是歷史兼容名稱。Cosmos runtime 實際只讀取：

| 欄位 | 功能 |
|---|---|
| `prompt` | 每次 observation 發送給 Cosmos 的任務指令 |
| `action_horizon` | 只作 `runtime.action_chunk_size` 缺省 fallback |

`ckpt`、`dataset_stats`、`train_config`、`mixed_precision`、`context_cache_dir`、`use_text_encoder`、`num_inference_steps` 等 FastWAM 模型欄位不會被遠端 Cosmos client 使用；Cosmos 模型配置在 server 端決定。

#### `runtime`

| 欄位 | 功能 |
|---|---|
| `action_chunk_size` | client 從每個 server action chunk 保留多少步，應為 32 |
| `rospy_rate` | 控制循環頻率，不是模型推理頻率 |
| `replan_interval` | 每多少個 control steps 發起一次新推理 |
| `exp_weight_factor` | 重疊 action chunks 的時間集成權重；越大越偏重新 chunk |
| `execute_actions` | `true` 才會向真機發命令 |
| `move_to_initial` | 啟動推理前是否移到配置的初始姿態 |
| `no_robot` | 跳過 CAN/pyAgxArm，使用模擬 state；仍然需要 ROS 三路相機 |
| `max_steps` | 最大控制步數，`null` 表示持續運行 |
| `rtc_debug` | 打印 RTC buffer 詳細信息 |
| `output_dir` | 保存 selected actions 的目錄 |

例如端到端約 5 秒、控制頻率 30 Hz 時，`replan_interval=8` 會嘗試每 0.267 秒重規劃，但目前 runtime 的推理呼叫是同步阻塞的，因此控制循環會被 inference latency 阻塞。不要把 `rospy_rate` 誤解為實際可達的閉環命令頻率。

#### `robot`

| 欄位 | 功能 |
|---|---|
| `left_channel` / `right_channel` | 左右 Piper CAN interface |
| `bitrate` | CAN bitrate，當前為 1,000,000 |
| `speed_pct` | Piper joint command speed percentage |
| `left_init_position` / `right_init_position` | 每側 6 joints + gripper 初始姿態 |
| `action_safety_threshold` | 相鄰 action 的平均絕對跳變上限 |
| `state_safety_threshold` | 當前 state 相對上一命令的平均追蹤誤差上限 |
| `max_action_delta` | 可選，單一 joint 最大 action 跳變 |
| `max_state_tracking_delta` | 可選，單一 joint 最大追蹤誤差 |
| `action_lower_bounds` / `action_upper_bounds` | 可選，14 維 action 絕對上下界 |

#### `cameras` 與 `ros_topics_dual`

Cosmos Piper14 強制需要三路 RGB：

```text
cam_high
cam_left_wrist
cam_right_wrist
```

`cameras` 中四個 enable flag 必須為 true，三個 camera name 必須精確匹配以上名稱。`ros_topics_dual` 指定三路 ROS `sensor_msgs/Image` topic。當前模型不使用 depth。

### 3.5 真機 client 啟動順序

先做不連機械臂測試，但保留 ROS 相機輸入：

```bash
source /opt/ros/noetic/setup.bash
cd /home/agilex/World_Action_Model/physical_WM/src/piper-cosmos3
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export COSMOS_PIPER14_CLIENT_TIMING=1

python3 scripts/run_real_cosmos_piper14_runtime.py \
  --config configs/real_deploy_cosmos_battery_motion.yaml \
  --host <SERVER_IP> \
  --port 8766 \
  --authkey cosmos-piper14 \
  --no-robot \
  --no-execute-actions \
  --no-move-to-initial \
  --max-steps 3 \
  --control-hz 30 \
  --replan-interval 8
```

再做連接雙臂但不執行 action 的測試：

```bash
python3 scripts/run_real_cosmos_piper14_runtime.py \
  --config configs/real_deploy_cosmos_battery_motion.yaml \
  --host <SERVER_IP> \
  --port 8766 \
  --authkey cosmos-piper14 \
  --no-execute-actions \
  --no-move-to-initial \
  --max-steps 3
```

只有在 metadata、相機、state、action shape、NaN/Inf 和安全閾值全部驗證通過後，才執行真機：

```bash
python3 scripts/run_real_cosmos_piper14_runtime.py \
  --config configs/real_deploy_cosmos_battery_motion.yaml \
  --host <SERVER_IP> \
  --port 8766 \
  --authkey cosmos-piper14 \
  --execute-actions \
  --move-to-initial
```

真機命令啟動後仍會等待一次 Enter 確認。首次驗證不要使用自動跳過安全確認的方式。

### 3.6 Client CLI 覆蓋規則

`run_real_cosmos_piper14_runtime.py` 的 CLI 會覆蓋 YAML：

| CLI | 覆蓋 YAML |
|---|---|
| `--host` | `policy_server.host` |
| `--port` | `policy_server.port` |
| `--authkey` | `policy_server.authkey` |
| `--execute-actions` / `--no-execute-actions` | `runtime.execute_actions` |
| `--move-to-initial` / `--no-move-to-initial` | `runtime.move_to_initial` |
| `--no-robot` | `runtime.no_robot=true` |
| `--max-steps` | `runtime.max_steps` |
| `--control-hz` | `runtime.rospy_rate` |
| `--replan-interval` | `runtime.replan_interval` |
| `--output-dir` | `runtime.output_dir` |
| `--prompt` | `fastwam.prompt` |

目前 `--port` 和 `--replan-interval` 都有非空 CLI 默認值，因此即使不寫它們，也會分別覆蓋 YAML 為 `8766` 和 `8`。為避免歧義，部署命令中應始終明確提供這兩個參數。

### 3.7 推薦啟動順序總結

```text
1. Server: source env，固定 GPU/offload，啟動 policy server
2. Client: metadata RPC smoke test
3. Client: HDF5 三次推理，不執行真機
4. Client: ROS + no_robot，不執行 action
5. Client: 連接雙臂，execute_actions=false
6. 人工確認初始姿態、CAN、相機與安全閾值
7. Client: execute_actions=true
```
