# Dry run TimeFM with Jane Street (jane-street-real-time-market-data-forecasting)

## Zero-shot TimeFM

* Use 3% `symbol_id` to dry run backtest.
    * data:
        * test_partitions: `8` and sampling another 3% `symbol_id` to be dry run test data
    * metrices:
        * backtest: `val_weighted_R2`: `0.8076`, Sharpe-like(annualized): `14.18`

## TimeFM LORA fine-tune

* Use 3% `symbol_id` to dry run fine-tune pipeline.
    * fine-tune model:
        * `jane_street_lora_adapter`: autoregression LORA.
        * `feature_head`: feature columns (00-78) use 2-layers MLP.
    * data:
        * train_partitions: `0`
        * val_partitions: `1`
        * test_partitions: `8` and sampling another 3% `symbol_id` to be dry run test data
    * metrices:
        * epoch 1: `val_weighted_R2`: `0.8270`, train_time: `17.12`  evaluate_time: `6:40`
        * epoch 2: `val_weighted_R2`: `0.8279`, train_time: `20:34`  evaluate_time: `7:21`
        * backtest: `val_weighted_R2`: `0.8263`, Sharpe-like(annualized): `14.19`

## Use dry run mode to check external regressors (XREG, features 00-78)

* Use 3% `symbol_id` to dry run fine-tune pipeline.
    * fine-tune model:
        * `jane_street_lora_adapter`: autoregression LORA.
        * `feature_head`: feature columns (00-78) use 2-layers MLP.
    * data:
        * train_partitions: `0`
        * val_partitions: `1`
        * test_partitions: `8` and sampling another 3% `symbol_id` to be dry run test data
    * metrices:
        * epoch 1: `val_weighted_R2`: `0.8274`, train_time: `15:30`  evaluate_time: `5:09`
        * epoch 2: `val_weighted_R2`: `0.8282`, train_time: `15:14`  evaluate_time: `5:21`
        * backtest: `val_weighted_R2`: `0.8255`, Sharpe-like(annualized): `14.18`
