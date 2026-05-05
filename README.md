# Data Download & Extraction Guide: Photometric Redshift Estimation

This guide explains how to set up your environment, download the raw astronomical FITS files
from the Sloan Digital Sky Survey (SDSS DR17), and process them into the final `.npy` arrays
required for training the deep learning models.

> **Prefer to skip the download?**  
> The fully processed `.npy` arrays are available directly on Google Drive:  
> [Download processed data (train/test .npy files)](https://drive.google.com/drive/folders/1aXbrHjE9N9tuHt3DQ-Ma_8TGXRQDVDxE?usp=sharing)

---

## 1. Prerequisites

Ensure the following are installed before starting.

**System tools**

- `rsync` — used to download FITS files from the SDSS Science Archive Server.
  Native on Linux and macOS. Windows users should use WSL or Git Bash.

**Python 3.8+**

Install all required Python packages with:

```bash
pip install numpy pandas astropy scipy tqdm
```

> `concurrent.futures` is part of the Python standard library and does not need to be installed separately.

**Storage**

Make sure you have sufficient free disk space before starting:

| Stage | Approximate Size |
|---|---|
| Raw `.fits.bz2` downloads | ~150–200 GB |
| Interim batch `.npy` files | ~4–5 GB |
| Final `.npy` arrays | ~4 GB |

---

## 2. Required Directory Structure

Before starting, your project directory must look exactly like this.
Create the `data/raw_fits/` folder manually — all other `data/` subdirectories
are created automatically by the scripts.

```text
Redshift-Estimation/
│
├── train.csv                         # Source catalogue (train set, 69,571 galaxies)
├── test.csv                          # Source catalogue (test set, 29,817 galaxies)
├── download_rsync.txt                # List of FITS files to download via rsync
├── download_and_process_data.py      # Parallel extraction script
├── merge_final.py                    # Script to merge interim batches into final arrays
│
└── data/
    └── raw_fits/                     # ← CREATE THIS FOLDER MANUALLY
                                      #   rsync will download all FITS files here,
                                      #   preserving the SDSS subdirectory structure:
                                      #   data/raw_fits/eboss/photoObj/frames/{rerun}/{run}/{camcol}/
```

The following subdirectories are created **automatically** by the scripts and do not need
to be created manually:

```text
data/
├── raw_fits/                         # (you create this)
├── interim_batches/                  # created by download_and_process_data.py
└── extracted_data/
    └── final/                        # created by merge_final.py
```

---

## 3. Understanding the Source Catalogues

`train.csv` and `test.csv` contain one row per galaxy. A representative row looks like:

```csv
objID,run,ra,dec,rerun,camcol,field,z,zErr,template_photo_z,template_photo_zErr,bin
1237651191892804003,1331,125.886104,48.547096,301,5,156,0.5939647,0.0001758,0.615261,0.058719,4
```

The key columns used by the pipeline are:

| Column | Description |
|---|---|
| `objID` | Unique SDSS object identifier — used to track processed galaxies |
| `run`, `camcol`, `field` | Identify the SDSS imaging field frame on disk |
| `rerun` | SDSS pipeline rerun number — used in the download path |
| `ra`, `dec` | Sky coordinates — used to locate the galaxy within the frame |
| `z` | Spectroscopic redshift (ground truth label) |
| `zErr` | Measurement error on spectroscopic redshift |
| `template_photo_z` | SDSS template-based photo-z estimate (baseline comparison) |
| `template_photo_zErr` | Error on the template photo-z |

The `bin` column is not used by any script.

---

## 4. Step-by-Step Process

### Step 1 — Verify your catalogues

Confirm that `train.csv` and `test.csv` are in the project root directory and contain
the columns listed above.

---

### Step 2 — Download raw FITS files via rsync

The `download_rsync.txt` file contains the exact paths of all required FITS files on the
SDSS server. It contains **115,885 lines**, corresponding to **23,177 unique field frames × 5
photometric bands** (u, g, r, i, z). Multiple galaxies from the catalogue can appear on the
same field frame, so the actual number of files to download is much smaller than the number
of galaxies.

Each path in `download_rsync.txt` follows this format:

```
eboss/photoObj/frames/{rerun}/{run}/{camcol}/frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2
```

Run the following command from the project root directory to start the download.
Replace `data/raw_fits/` with a different path only if you have a specific storage location in mind
(but then remember to use that same path in Step 3):

```bash
rsync -avz --files-from=download_rsync.txt rsync://data.sdss.org/dr17/ data/raw_fits/
```

**Important notes:**

- `rsync` is resumable. If your connection drops, run the exact same command again.
  It will verify existing files and download only what is missing.
- The downloaded files remain **compressed** (`.fits.bz2`). Do **not** manually decompress them.
  The extraction script reads `.bz2` files directly using `astropy`, which handles
  decompression automatically in memory.
- The download may take several hours to days depending on your network speed and
  the SDSS server load.

---

### Step 3 — Extract galaxy image cutouts

Once the FITS files are downloaded, run the parallel extraction script. You **must** pass
`--search_dir` pointing to the same folder where rsync placed the files:

```bash
python download_and_process_data.py --search_dir data/raw_fits/
```

**Optional arguments:**

| Argument | Default | Description |
|---|---|---|
| `--search_dir` | `data/decompressed_files` | Path to the folder containing the downloaded `.fits.bz2` files. **Must match the rsync destination from Step 2.** |
| `--save_dir` | `data/interim_batches` | Path where extracted `.npy` chunk files are saved. |

**What this script does:**

1. Reads `train.csv` and `test.csv` to get the full list of galaxies.
2. Scans `--search_dir` recursively to find which field frames have been downloaded.
3. Skips any galaxy whose `objID` already appears in a previously saved chunk file
   (smart resume — safe to restart if interrupted).
4. For each galaxy, it:
   - Locates the five band FITS files (u, g, r, i, z) for that field on disk.
   - Opens each `.fits.bz2` file directly using `astropy` (no manual decompression needed).
   - Applies `BSCALE`/`BZERO` calibration from the FITS header.
   - Uses the WCS header to convert the galaxy's (RA, Dec) to pixel coordinates.
   - Extracts a 40×40 pixel cutout centred on the galaxy, zero-padding if near a detector edge.
   - Stacks the five bands into a single `(40, 40, 5)` array.
5. Saves results to `--save_dir` in chunks of 2,000 galaxies:
   `X_part_N.npy`, `y_part_N.npy`, `ids_part_N.npy`.
6. After processing, generates `download_rsync_remaining.txt` listing any galaxies
   whose FITS files were not yet downloaded, ready for a follow-up rsync run.

The script uses all available CPU cores via `ProcessPoolExecutor`. It can be safely
interrupted and restarted at any point — already-processed galaxies are skipped automatically.

---

### Step 4 — Merge interim batches into final arrays

Once all chunks are saved in `data/interim_batches/`, run:

```bash
python merge_final.py
```

**What this script does:**

1. Loads all `X_part_*.npy`, `y_part_*.npy`, and `ids_part_*.npy` chunk files.
2. Uses the `objID` values to cross-reference with `train.csv` and `test.csv`,
   strictly reconstructing the original train/test split to prevent data leakage.
3. Saves four final files to `data/extracted_data/final/`.

---

## 5. Final Verification

After Step 4 completes, verify the output:

```bash
python -c "
import numpy as np
X_train = np.load('data/extracted_data/final/X_train.npy')
y_train = np.load('data/extracted_data/final/y_train.npy')
X_test  = np.load('data/extracted_data/final/X_test.npy')
y_test  = np.load('data/extracted_data/final/y_test.npy')
print('X_train:', X_train.shape)   # expected: (69571, 40, 40, 5)
print('y_train:', y_train.shape)   # expected: (69571, 4)
print('X_test: ', X_test.shape)    # expected: (29817, 40, 40, 5)
print('y_test: ', y_test.shape)    # expected: (29817, 4)
"
```

Expected output:

```
X_train: (69571, 40, 40, 5)
y_train: (69571, 4)
X_test:  (29817, 40, 40, 5)
y_test:  (29817, 4)
```

The four columns in `y_train` and `y_test` correspond to:
`[z, zErr, template_photo_z, template_photo_zErr]`

You are now ready to run the model training scripts (e.g., `denseNet_1.py`, `pasquet_inception_1.py`).

---

## 6. Troubleshooting

**The extraction script finds 0 galaxies to process**

The most common cause is a mismatch between the rsync download destination and `--search_dir`.
Make sure both point to the same folder. For example, if you ran:

```bash
rsync ... rsync://data.sdss.org/dr17/ /mnt/storage/raw_fits/
```

then you must run:

```bash
python download_and_process_data.py --search_dir /mnt/storage/raw_fits/
```

**rsync fails or is very slow**

The SDSS server can be slow during peak hours. The `-z` flag enables compression during
transfer which helps on slow connections. If the server times out, simply re-run the
same rsync command — it will resume from where it left off.

**Some galaxies are missing from the final arrays**

If fewer galaxies than expected appear in the final arrays, it means their FITS files
were not downloaded. After running the extraction script, check
`download_rsync_remaining.txt` in the project root — it lists all missing files.
Run a second rsync pass using:

```bash
rsync -avz --files-from=download_rsync_remaining.txt rsync://data.sdss.org/dr17/ data/raw_fits/
```

Then re-run Steps 3 and 4. The smart resume feature ensures already-processed galaxies
are not re-extracted.
