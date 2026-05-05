import glob
import numpy as np
import pandas as pd
import os

def check_duplicates():
    print("--- 1. CHECKING EXTRACTED BATCHES FOR DUPLICATE IDs ---")
    batch_dir = "data/interim_batches"
    
    # Load all ids_part_*.npy files
    id_files = glob.glob(os.path.join(batch_dir, "ids_part_*.npy"))
    
    if not id_files:
        print("Could not find interim batches. Make sure paths are correct.")
    else:
        all_extracted_ids = []
        for f in id_files:
            all_extracted_ids.extend(np.load(f))
            
        total_extracted = len(all_extracted_ids)
        unique_extracted = len(set(all_extracted_ids))
        duplicates_count = total_extracted - unique_extracted
        
        print(f"Total galaxies extracted: {total_extracted}")
        print(f"Unique galaxies: {unique_extracted}")
        if duplicates_count > 0:
            print(f"⚠️ WARNING: Found {duplicates_count} duplicate galaxies in your downloaded batches!")
        else:
            print("✅ No duplicates found in downloaded batches.")


    print("\n--- 2. CHECKING ORIGINAL CSVs FOR TRAIN/TEST LEAKAGE ---")
    # Load the Starkindler CSVs
    try:
        train_df = pd.read_csv("train.csv")
        test_df = pd.read_csv("test.csv")
        
        train_ids = set(train_df['objID'])
        test_ids = set(test_df['objID'])
        
        # Check intersections (Galaxies in BOTH train and test)
        leakage = train_ids.intersection(test_ids)
        
        print(f"Total IDs in train.csv: {len(train_ids)}")
        print(f"Total IDs in test.csv: {len(test_ids)}")
        
        if len(leakage) > 0:
            print(f"⚠️ CRITICAL DATA LEAKAGE: Found {len(leakage)} galaxies that are in BOTH train.csv and test.csv!")
        else:
            print("✅ Train and Test sets are perfectly separated (No leakage).")
            
        # Check for duplicates WITHIN train.csv
        if len(train_df) != len(train_ids):
            print(f"⚠️ WARNING: train.csv contains {len(train_df) - len(train_ids)} duplicate rows internally!")
        else:
            print("✅ No internal duplicates in train.csv.")

    except FileNotFoundError:
        print("Could not find train.csv or test.csv in the current directory.")

if __name__ == "__main__":
    check_duplicates()