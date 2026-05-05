import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# Setup Paths
SCRIPT_DIR = Path(__file__).parent
BATCH_DIR = SCRIPT_DIR / "data/interim_batches"
FINAL_DIR = SCRIPT_DIR / "data/extracted_data/final"

def merge_and_split():
    print("Step 1: Loading all batch parts...")
    # Find all parts
    x_files = sorted(glob.glob(str(BATCH_DIR / "X_part_*.npy")))
    y_files = sorted(glob.glob(str(BATCH_DIR / "y_part_*.npy")))
    id_files = sorted(glob.glob(str(BATCH_DIR / "ids_part_*.npy")))

    if not x_files:
        print("Error: No batch files found in 'data/interim_batches'.")
        return

    # Concatenate everything into one massive pile
    print(f"Merging {len(x_files)} chunks...")
    full_X = np.concatenate([np.load(f) for f in x_files], axis=0)
    full_y = np.concatenate([np.load(f) for f in y_files], axis=0)
    full_ids = np.concatenate([np.load(f) for f in id_files], axis=0)

    print(f"Total Processed Galaxies: {len(full_ids)}")

    # Step 2: Split into Train and Test based on your original CSVs
    print("Step 2: Splitting into Train/Test...")
    train_df = pd.read_csv(SCRIPT_DIR / "train.csv")
    test_df = pd.read_csv(SCRIPT_DIR / "test.csv")

    # Create a lookup map: ObjID -> Index in full array
    # This lets us find where galaxy '123' is located in the massive array
    id_map = {obj_id: idx for idx, obj_id in enumerate(full_ids)}

    # Find indices for Train and Test sets
    train_indices = []
    for uid in train_df['objID']:
        if uid in id_map:
            train_indices.append(id_map[uid])
            
    test_indices = []
    for uid in test_df['objID']:
        if uid in id_map:
            test_indices.append(id_map[uid])

    # Step 3: Save Final Files
    os.makedirs(FINAL_DIR, exist_ok=True)

    print(f"Saving Train Set ({len(train_indices)} samples)...")
    np.save(FINAL_DIR / "X_train.npy", full_X[train_indices])
    np.save(FINAL_DIR / "y_train.npy", full_y[train_indices])

    print(f"Saving Test Set ({len(test_indices)} samples)...")
    np.save(FINAL_DIR / "X_test.npy", full_X[test_indices])
    np.save(FINAL_DIR / "y_test.npy", full_y[test_indices])

    print("\nSUCCESS! Your data is ready.")
    print(f"Location: {FINAL_DIR}")

if __name__ == "__main__":
    merge_and_split()