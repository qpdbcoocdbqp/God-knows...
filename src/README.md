# TimesFM 2.5 (LoRA) × Jane Street Pipeline

> **第二次更新：訓練改成官方文件的 LoRA 微調方式。**
> 參考 [`google-research/timesfm` 的官方微調範例](https://github.com/google-research/timesfm/tree/d720daa6786539c2566a44464fbda1019c0a82c0/timesfm-forecasting/examples/finetuning)
> (`finetune_lora.py`) 重寫了訓練相關程式碼。`timesfm` 這個獨立套件**完全沒有公開的微調介面**；
> 官方做法是改用 **HuggingFace `transformers` 的 `TimesFm2_5ModelForPrediction`** + **`peft` 的 LoRA**。

## 官方參考腳本重點（已實際下載原始碼確認）

```python
from peft import LoraConfig, get_peft_model
from transformers import TimesFm2_5ModelForPrediction

model = TimesFm2_5ModelForPrediction.from_pretrained(
    "google/timesfm-2.5-200m-transformers",
    torch_dtype=torch.bfloat16,
    device_map=device,
)
model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules="all-linear",
                                          lora_dropout=0.05, bias="none"))

outputs = model(past_values=context, future_values=target, forecast_context_len=context_len)
loss = outputs.loss                 # 官方內建（未加權）loss
pred = outputs.mean_predictions     # (batch, horizon) 點預測
```

- 注意：這裡用的是 **transformers 版 checkpoint**（`google/timesfm-2.5-200m-transformers`），
  跟之前用 `timesfm` 套件本身的 `google/timesfm-2.5-200m-pytorch` 不是同一個 repo id。
- `past_values` 不需要外部 normalize，模型內部自己做 RevIN（官方文件特別強調 "Do not normalise
  your data externally"）。
- 官方 `outputs.loss` 是**未加權**的 MSE + quantile loss，且是在 normalize 過的空間計算。
  Jane Street 評分是用官方的 `weight` 欄位加權，所以本 pipeline **沒有使用 `outputs.loss`**，
  而是另外用 `outputs.mean_predictions` 對齊真實值，套用自訂的 `JaneStreetWeightedMSELoss`。

## 這次改了什麼

| 檔案 | 改動 |
|---|---|
| `model.py` | 整個重寫：改用 `transformers.TimesFm2_5ModelForPrediction.from_pretrained(...)` + `peft.LoraConfig`/`get_peft_model`/`PeftModel`，拿掉先前對 `timesfm` 套件內部 `revin`/`update_running_stats` 的手動重建邏輯（不再需要）。 |
| `train.py` | 訓練迴圈改成呼叫 `model(past_values=, future_values=, forecast_context_len=)`，loss 換成自訂加權 loss；以 validation 加權 R² 做 early stopping，最佳結果用 `model.save_pretrained(...)` 存 LoRA adapter（而不是存整個模型 state_dict）。 |
| `data.py` | 新增官方風格的 `TimeSeriesRandomWindowDataset`（訓練用，隨機抽樣 context/horizon 窗口，比固定窗口更省資料），`TimeSeriesLastWindowDataset` 用於 val/test 的確定性評估；兩者都帶上 Jane Street 的 `weight` 欄位（官方範例本身沒有 weight 概念，這裡是擴充）。 |
| `backtest.py` | 改成載入 `PeftModel`（fine-tuned adapter）與原始 base model 分別跑 `model(past_values=...)`，比較 zero-shot vs. LoRA 微調後的加權 R² / PnL，呼應官方腳本自帶的 `evaluate()` zero-shot 對照邏輯。 |
| `config.py` | `ModelConfig` 改成 `model_id`（transformers checkpoint）、`context_len`、`horizon_len`、`torch_dtype`；新增 `LoraConfigOpts`（r/alpha/dropout/target_modules/bias，預設值對齊官方腳本）；`TrainConfig.output_dir` 現在是 adapter 目錄。 |
| `main.py` | CLI 從 `--checkpoint`（state_dict 路徑）改成 `--adapter-dir`（LoRA adapter 目錄），呼應 `model.save_pretrained` / `PeftModel.from_pretrained` 的存取方式。 |
| `requirements.txt` | 拿掉 `timesfm`，改成 `transformers` + `peft` + `accelerate`。 |
| `losses.py` | 未變動，`JaneStreetWeightedMSELoss` 直接套用在 `outputs.mean_predictions` 上即可。 |

## 檔案結構

```
jane_street_timesfm/
├── config.py      # DataConfig / ModelConfig / LoraConfigOpts / TrainConfig / BacktestConfig
├── data.py        # 隨機窗口訓練集 + 滑動窗口評估集（含 weight）
├── losses.py      # JaneStreetWeightedMSELoss（主要使用）+ 修正過的 JaneStreetMultitaskLoss（多目標擴充用）
├── model.py        # transformers + peft LoRA 模型載入/套用/存取
├── train.py        # LoRA 微調迴圈：加權 loss + 加權 R² early stopping
├── backtest.py     # 加權 R²、PnL/equity curve、zero-shot vs. LoRA 對照
├── main.py         # CLI 入口：--stage train / backtest / all
└── requirements.txt
```

## 資料假設

`DataConfig.data_dir` 指向 Jane Street "Real-Time Market Data Forecasting" 的 parquet 目錄：

```
train.parquet/
├── partition_id=0/part-0.parquet
├── partition_id=1/part-0.parquet
...
```

需要欄位：`date_id`, `time_id`, `symbol_id`, `weight`, `responder_6`
（可在 `DataConfig.target_col` 改成其他 responder）。

## 安裝

```bash
pip install -r requirements.txt
```

## 使用方式

```bash
# LoRA 微調 + 回測一次跑完（含 zero-shot 對照）
python main.py --stage all

# 只訓練
python src/main.py --stage train

# 只用已存的 LoRA adapter 回測
python main.py --stage backtest --adapter-dir ./checkpoints/jane_street_lora_adapter --split test

# 跳過 zero-shot 對照（只跑微調後模型）
python main.py --stage backtest --no-zero-shot
```

## 重要提醒

- 本次重寫的 API 呼叫已對照**實際安裝**的 `transformers`（含 `models/timesfm2_5/modeling_timesfm2_5.py`）
  原始碼驗證過：`forward()` 的參數名稱、`outputs.loss` / `outputs.mean_predictions` 欄位、
  `model.config.context_length` 都已確認存在；`finetune_lora.py` 原始碼也已下載核對過完整流程
  （資料集設計、LoRA config、訓練/評估迴圈）。
- 沒有用真實 Jane Street 資料 + 真實下載的預訓練權重端到端跑過一次完整訓練（受限於目前驗證環境的
  網路白名單，無法連線 huggingface.co 下載權重）。如果你環境能連線 HuggingFace，理論上可以直接跑；
  若 shape 或欄位名稱對不上，最可能是你安裝的 `transformers` 版本與這裡驗證的版本（5.12.1）有差異。
- `torch_dtype="bfloat16"` 是跟官方範例一致的預設值；純 CPU 環境建議在 `config.py` 把
  `ModelConfig.torch_dtype` 改成 `"float32"`。
- 若想對照原始 `trains.py` 片段中 `JaneStreetMultitaskLoss`（多 responder 同時預測）的概念，
  `losses.py` 仍保留一份修正過的版本，可在擴充成多輸出 head 時使用；目前主線程式碼用的是單一
  target（`responder_6`）的 `JaneStreetWeightedMSELoss`。

---

## 第三次更新：資料讀取改成 pyarrow.dataset lazy batched streaming

`data.py` 整個重寫，拿掉所有 pandas（`pd.read_parquet` / `pd.concat` /
groupby 整批切 per-symbol 序列），改用：

- `pyarrow.dataset.dataset(data_dir, format="parquet", partitioning="hive")`
  建立 Dataset（**不會**在這一步讀任何資料，只掃 schema/檔案列表）。
- `dataset.scanner(columns=[...], filter=ds.field("partition_id").isin([...]), batch_size=...)`：
  只投影需要的 5 個欄位、用 partition filter 做 partition pruning（只開需要的 partition 檔案），
  並用 `scanner.to_batches()` **逐批（lazy generator）**讀取，而不是一次整批讀進記憶體。
- 每讀進一批 `RecordBatch`，逐 row 更新每個 `symbol_id` 各自的
  `collections.deque(maxlen=context_len+horizon_len)` 滑動窗口；窗口一滿就直接組成
  `(context, target, weight)` 並轉成 torch tensor 吐出，因此**整個訓練/驗證資料集**
  在記憶體中只會同時存在「symbol 數量 × 窗口長度」這麼小的一塊資料，不管 partition 檔案實際多大。
- 訓練集用「streaming shuffle buffer」（reservoir-style 隨機交換）做局部打亂，取代之前
  「先抽樣 N 個固定隨機窗口」的做法；一個 epoch = 對 train partitions 完整掃過一輪。
- 因為現在是 `torch.utils.data.IterableDataset`（沒有 `len()`），`train.py` 的 LR scheduler
  從 `CosineAnnealingLR`（需要預知總步數）改成 `ReduceLROnPlateau`（依 validation 加權 R² 調整）。
- 支援 `DataLoader(num_workers>0)`：會自動把 partition 清單依 worker 數量切開，避免每個
  worker 重複掃同一份資料。

### 新增的 `DataConfig` 欄位

| 欄位 | 預設 | 說明 |
|---|---|---|
| `arrow_batch_size` | `50_000` | pyarrow 每次從磁碟讀出的 row 數（不是訓練 batch size）|
| `train_stride` | `1` | 每個 symbol 隔幾個新 row 才吐出一個訓練窗口（1 = 最大重疊利用率）|
| `eval_stride` | `None`（= horizon_len）| val/test 窗口間隔，預設不重疊 |
| `shuffle_buffer_size` | `4096` | streaming shuffle buffer 大小 |

### 已驗證

用合成的 hive-partitioned parquet 測試集實際跑過：欄位投影、partition filter
pruning、`to_batches()` streaming、滑動窗口正確性（窗口數 = `每symbol列數 - window_len + 1`，
已用兩個 symbol 各 500 列驗證算出來剛好對得上）、`num_workers>0` 的 partition 切分、以及
`backtest.py` 用假 model 跑過完整 `generate_predictions` → `simulate_pnl` 流程，全部正常。

如果你的資料*不是*依 `date_id, time_id` 大致排序存放（理論上 Jane Street 官方 parquet 是這樣排的），
滑動窗口的時間順序假設會不成立；可以之後加一個「每批內先排序」的選項（仍然是 streaming，只排
單一 batch，不影響整體 lazy 特性）。
