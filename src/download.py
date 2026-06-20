import kaggle

# Authenticate using your kaggle.json token
kaggle.api.authenticate()

# Download a community-uploaded version of the Jane Street dataset
# This is a standard "Dataset" and does NOT require phone verification or rule acceptance!
dataset_name = 'mohamedsameh0410/jane-street-dataset'

print(f"Downloading dataset {dataset_name}...")
kaggle.api.dataset_download_files(dataset_name, path='.', unzip=True)
print("Download complete!")
