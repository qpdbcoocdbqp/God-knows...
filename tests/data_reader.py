import pandas as pd
import pyarrow.dataset as ds


def load_dataset():
    # 1. 讀取 features.csv (特徵與標籤元數據)
    features_path = 'data/jane-street-real-time-market-data-forecasting/features.csv'
    features_df = pd.read_csv(features_path)
    print(f"Loaded features metadata: {features_df.shape}")

    # 2. 讀取 lags.parquet (前一時間步目標變數)
    lags_path = 'data/jane-street-real-time-market-data-forecasting/lags.parquet'
    lags_df = pd.read_parquet(lags_path)
    print(f"Loaded lags data: {lags_df.shape}")

    # 3. 讀取 train.parquet (分割區 Parquet 數據集)
    # 注意：train.parquet 非常巨大，建議使用 pyarrow.dataset 進行延遲加載或只篩選讀取特定分割區，以避免記憶體溢出
    train_path = 'data/jane-street-real-time-market-data-forecasting/train.parquet'
    train_dataset = ds.dataset(train_path, format="parquet", partitioning="hive")
    print("Train dataset initialized (lazy loading).")
    num_rows = train_dataset.count_rows()
    print(f"Train dataset has {num_rows * 1e-6:.2f} million rows.")
    print("Schema contains", len(train_dataset.schema), "columns.")

    # 範例：如何讀取一部分 train 資料 (例如 partition_id = 0)
    first_partition = train_dataset.to_table(filter=ds.field('partition_id') == 0)
    print(f"Loaded train partition 0: {first_partition.shape}")

    return features_df, lags_df, train_dataset

if __name__ == '__main__':
    load_dataset()

