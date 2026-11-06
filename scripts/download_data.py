"""
Downloads a community-uploaded copy of the Jane Street dataset from Kaggle.

It is a standard Kaggle "Dataset", so it needs a `kaggle.json` token but no phone
verification or competition rule acceptance.

    python scripts/download_data.py
"""

DATASET_NAME = "mohamedsameh0410/jane-street-dataset"


def main() -> None:
    import kaggle

    kaggle.api.authenticate()
    print(f"Downloading dataset {DATASET_NAME}...")
    kaggle.api.dataset_download_files(DATASET_NAME, path=".", unzip=True)
    print("Download complete!")


if __name__ == "__main__":
    main()
