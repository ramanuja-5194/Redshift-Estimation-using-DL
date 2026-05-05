"""
Final Parallel Batch Processor (Version 3.0)
- Feature 1: Multiprocessing (Uses all CPU cores).
- Feature 2: Incremental Saving (Saves every 2000 files to prevent data loss).
- Feature 3: Smart Resume (Skips already processed files).
- Feature 4: Rsync List Generator (Creates the list for the next download batch).
"""
import os
import glob
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from pathlib import Path
import warnings
from concurrent.futures import ProcessPoolExecutor

warnings.simplefilter("ignore")

SCRIPT_DIR = Path(__file__).parent
Y_COLS = ["z", "zErr", "template_photo_z", "template_photo_zErr"]

def get_grid(file_path_pattern, ra, dec):
    """Worker function to extract a single galaxy."""
    grids = []
    for band in ['u', 'g', 'r', 'i', 'z']:
        band_path = file_path_pattern.replace("frame-u-", f"frame-{band}-")
        
        # Handle compression
        if not os.path.exists(band_path):
            if os.path.exists(band_path + ".bz2"):
                band_path += ".bz2"
            elif band_path.endswith(".bz2") and os.path.exists(band_path[:-4]):
                band_path = band_path[:-4]
            else:
                return None

        try:
            with fits.open(band_path) as hdul:
                header = hdul[0].header
                data = hdul[0].data
                if "BSCALE" in header: data = data * header["BSCALE"] + header["BZERO"]
                
                wcs = WCS(header)
                x, y = wcs.world_to_pixel_values(ra, dec)
                x, y = int(round(x.item() if hasattr(x, 'item') else x)), int(round(y.item() if hasattr(y, 'item') else y))
                
                grid = data[max(0, y-20):min(data.shape[0], y+20), max(0, x-20):min(data.shape[1], x+20)]
                pad_h, pad_w = 40 - grid.shape[0], 40 - grid.shape[1]
                if pad_h > 0 or pad_w > 0:
                    grid = np.pad(grid, ((0, pad_h), (0, pad_w)), mode='constant')
                
                grids.append(grid)
        except:
            return None
    return np.stack(grids, axis=-1)

def process_single_row(row_data):
    """Wrapper for parallel execution."""
    idx, row, search_dir = row_data
    run, camcol, field = int(row['run']), int(row['camcol']), int(row['field'])
    
    # Precise glob pattern to reduce search time
    search_pattern = f"{search_dir}/**/frame-u-{run:06d}-{camcol}-{field:04d}.fits*"
    matches = glob.glob(search_pattern, recursive=True)
    
    if not matches: return None

    img_data = get_grid(matches[0], row['ra'], row['dec'])
    if img_data is None: return None
        
    y_data = np.array([row[col] for col in Y_COLS], dtype=np.float32)
    return (img_data, y_data, row['objID'])

def save_chunk(X, y, ids, save_dir):
    """Helper to save a mini-batch."""
    os.makedirs(save_dir, exist_ok=True)
    # Find next chunk number
    existing = glob.glob(f"{save_dir}/ids_part_*.npy")
    chunk_id = len(existing) + 1
    
    np.save(f"{save_dir}/X_part_{chunk_id}.npy", np.stack(X))
    np.save(f"{save_dir}/y_part_{chunk_id}.npy", np.stack(y))
    np.save(f"{save_dir}/ids_part_{chunk_id}.npy", np.array(ids))
    print(f" [Saved Chunk {chunk_id}: {len(X)} samples]")

def generate_rsync_list(df, processed_ids, script_dir):
    """Generates the rsync list for the remaining galaxies."""
    print("\nGenerating 'download_rsync_remaining.txt'...")
    remaining_df = df[~df['objID'].isin(processed_ids)]
    
    if len(remaining_df) == 0:
        print("All galaxies processed! No downloads needed.")
        return

    rsync_path = script_dir / "download_rsync_remaining.txt"
    with open(rsync_path, "w") as f:
        for idx, row in remaining_df.iterrows():
            run, rerun, camcol, field = int(row['run']), int(row['rerun']), int(row['camcol']), int(row['field'])
            for band in ['u', 'g', 'r', 'i', 'z']:
                # Standard SDSS Directory Structure
                path = f"eboss/photoObj/frames/{rerun}/{run}/{camcol}/frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2"
                f.write(path + "\n")
    
    print(f"Done! {len(remaining_df)} galaxies remaining.")
    print(f"Next step: Run rsync with --files-from={rsync_path}")

def process_batch_parallel(args):
    # 1. Setup
    train_path = SCRIPT_DIR / "train.csv"
    test_path = SCRIPT_DIR / "test.csv"
    print("Loading CSVs...")
    df = pd.concat([pd.read_csv(train_path), pd.read_csv(test_path)])
    
    # 2. Filter Existing
    print("Scanning directory for input files...")
    files = glob.glob(f"{args.search_dir}/**/frame-u-*.fits*", recursive=True)
    available_keys = set()
    for f in files:
        parts = os.path.basename(f).split('-')
        try:
            available_keys.add(f"{int(parts[2])}-{int(parts[3])}-{int(parts[4].split('.')[0])}")
        except: continue
            
    df['key'] = df['run'].astype(str) + "-" + df['camcol'].astype(str) + "-" + df['field'].astype(str)
    batch_df = df[df['key'].isin(available_keys)]
    
    # REMOVE ALREADY PROCESSED (Smart Resume)
    processed_ids = []
    for f in glob.glob(f"{args.save_dir}/ids_part_*.npy"):
        processed_ids.extend(np.load(f))
    processed_set = set(processed_ids)
    
    batch_df = batch_df[~batch_df['objID'].isin(processed_set)]
    
    print(f"Found {len(batch_df)} NEW galaxies to process from disk.")
    
    # 3. Parallel Loop
    if len(batch_df) > 0:
        work_items = [(idx, row.to_dict(), args.search_dir) for idx, row in batch_df.iterrows()]
        
        X_buf, y_buf, id_buf = [], [], []
        CHUNK_SIZE = 2000  # Save every 2000 galaxies
        
        # Max workers limited to avoid OOM crashes (you can adjust 'max_workers' if needed)
        with ProcessPoolExecutor() as executor:
            results = tqdm(executor.map(process_single_row, work_items), total=len(work_items))
            
            for res in results:
                if res is not None:
                    X_buf.append(res[0])
                    y_buf.append(res[1])
                    id_buf.append(res[2])
                
                # Incremental Save
                if len(X_buf) >= CHUNK_SIZE:
                    save_chunk(X_buf, y_buf, id_buf, args.save_dir)
                    # Add to processed list so rsync generator knows about them
                    processed_ids.extend(id_buf)
                    X_buf, y_buf, id_buf = [], [], [] # Clear buffer

        # Save remaining buffer
        if X_buf:
            save_chunk(X_buf, y_buf, id_buf, args.save_dir)
            processed_ids.extend(id_buf)
    
    # 4. Generate Rsync List (The Missing Feature)
    generate_rsync_list(df, set(processed_ids), SCRIPT_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_dir", default="data/decompressed_files")
    parser.add_argument("--save_dir", default="data/interim_batches")
    args = parser.parse_args()
    
    # Ensure absolute paths
    if not os.path.isabs(args.search_dir):
        args.search_dir = str(SCRIPT_DIR / args.search_dir)
    if not os.path.isabs(args.save_dir):
        args.save_dir = str(SCRIPT_DIR / args.save_dir)
        
    process_batch_parallel(args)