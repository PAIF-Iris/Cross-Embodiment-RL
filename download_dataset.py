import tensorflow_datasets as tfds
import tqdm

# optionally replace the DATASET_NAMES below with the list of filtered datasets from the google sheet
DATASET_NAMES = ['robonet']
DOWNLOAD_DIR = '/Users/paif_iris/Desktop/metaworld/dataset-1'

print(f"Downloading {len(DATASET_NAMES)} datasets to {DOWNLOAD_DIR}.")

for dataset_name in tqdm.tqdm(DATASET_NAMES):
  _ = tfds.load(dataset_name, data_dir=DOWNLOAD_DIR)
print("All datasets downloaded.")