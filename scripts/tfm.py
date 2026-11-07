import torch
import gc

from jsforecast.config import PipelineConfig
from jsforecast.data import JaneStreetDataModule
from jsforecast.models import build_forecaster
from jsforecast.trainer import Trainer
from jsforecast.utils import seed_everything
from jsforecast.evaluate import Evaluator, Predictor


# config
MODEL_TYPE = "timesfm3"  # "timesfm2_5" or "timesfm3"
ADAPTER_DIR = "./checkpoints/timesfm3_lora_fast"  # where the adapter is saved / loaded
CONTEXT_LEN = 256  # speed test: same val R2 as 512, faster
HORIZON = 1  # the Jane Street task is next-step

cfg = PipelineConfig()
cfg.model.model_type = MODEL_TYPE
cfg.model.context_len = CONTEXT_LEN
cfg.model.horizon_len = HORIZON
cfg.train.output_dir = ADAPTER_DIR
cfg.model.use_xreg = False
cfg.model.symmetric = False  # speed test: ~38% faster, val R2 within 0.001; use the same setting to predict
cfg.data.symbol_sample_rate = 0.03
cfg.train.epochs = 1
cfg.train.batch_size = 32
cfg.train.max_train_batches = 1200
cfg.train.max_val_batches = 96  # used by the Predictor/Evaluator comparison below
cfg.train.validate = False  # no validation inside the Trainer
cfg.model.__post_init__()

# train adapter
seed_everything(cfg.data.random_seed)
trainee = build_forecaster(cfg.model, lora_cfg=cfg.lora, data_cfg=cfg.data, trainable=True)
datamodule = JaneStreetDataModule(cfg.data, trainee.context_len, cfg.model.horizon_len)
result = Trainer(trainee, datamodule, cfg.train, cfg.data).fit()
print("adapter saved to:", result.output_dir)

del trainee

# compare zero-shot and the adapter on the same validation windows
predictor = Predictor(cfg.data)
evaluator = Evaluator(cfg.data)
val_batch_size = cfg.train.eval_batch_size

scores = {}
for name, adapter_dir in {"zero_shot": None, "adapter": ADAPTER_DIR}.items():
    model = build_forecaster(cfg.model, adapter_dir=adapter_dir)
    loader = datamodule.loader("val", val_batch_size)  # fresh loader: same windows every time
    table = predictor.run(model, loader, max_batches=cfg.train.max_val_batches, desc=name)
    scores[name] = evaluator.score(table)
    del model
    gc.collect()
    torch.cuda.empty_cache()

for name, m in scores.items():
    print(f"{name:10s} R2={m.r2:.4f}  pnl={m.total_pnl:,.0f}  sharpe={m.sharpe:.2f}  days={m.n_days}  rows={m.n_rows:,}")

# test split: load saved adapters from checkpoints/ (no training) and score them on the same windows
TEST_ADAPTERS = {
    "zero_shot": None,
    "r4": ADAPTER_DIR,
}
TEST_MAX_BATCHES = 96  # 96 * eval_batch_size windows, same count as the validation runs

test_datamodule = JaneStreetDataModule(cfg.data, cfg.model.context_len, cfg.model.horizon_len)
test_predictor = Predictor(cfg.data)
test_evaluator = Evaluator(cfg.data)

test_scores = {}
for name, adapter_dir in TEST_ADAPTERS.items():
    model = build_forecaster(cfg.model, adapter_dir=adapter_dir)
    loader = test_datamodule.loader("test", cfg.train.eval_batch_size)
    table = test_predictor.run(model, loader, max_batches=TEST_MAX_BATCHES, desc=f"test {name}")
    test_scores[name] = test_evaluator.score(table)
    del model
    gc.collect()
    torch.cuda.empty_cache()

for name, m in test_scores.items():
    print(f"test {name:10s} R2={m.r2:.4f}  pnl={m.total_pnl:,.0f}  sharpe={m.sharpe:.2f}  days={m.n_days}  rows={m.n_rows:,}")
