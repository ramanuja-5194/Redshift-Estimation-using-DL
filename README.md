# Data Download & Extraction Guide: Photometric Redshift Estimation

This guide explains how to set up your directory, download the raw astronomical FITS files from the Sloan Digital Sky Survey (SDSS DR17), and process them into the final `.npy` arrays required for training the Deep Learning models.

## 1. Prerequisites

Before starting, ensure you have the following installed on your system:
*   **Python 3.8+**
*   **Required Python Packages:** `numpy`, `pandas`, `astropy`, `scipy`
*   **rsync:** Required for downloading the FITS files from the SDSS Science Archive Server (SAS). (Native on Linux/macOS; Windows users can use WSL or Git Bash).

## 2. Required Directory Structure

Before initiating the download and extraction process, your project directory must look like this. Create any missing folders manually:

```text
Redshift-Estimation/
│
├── train.csv                     # Source catalog (must contain objID, run, rerun, camcol, field, z)
├── test.csv                      # Source catalog for test set
├── download_and_process_data.py  # Script for parallel extraction
├── merge_final.py                # Script to merge interim batches
├── download_rsync.txt            # Generated list of FITS files to download via rsync
```

## 3. Step-by-Step Download and Extraction Process

### Step 1: Prepare the Target Catalogues
Ensure that `train.csv` and `test.csv` are placed in the root directory. These files contain the spectroscopic redshifts (ground truth) and the unique SDSS identifiers (`run`, `rerun`, `camcol`, `field`) needed to locate the exact FITS files on the SDSS servers.

### Step 2: Download Raw FITS Files using `rsync`
Because downloading ~500,000 FITS files (5 bands for ~100,000 galaxies) is highly susceptible to network interruptions, we use `rsync`. 

1. Ensure your `download_rsync.txt` file is generated. This file contains the exact paths to the SDSS server in the format:
   `eboss/photoObj/frames/{rerun}/{run}/{camcol}/frame-{band}-{run}-{camcol}-{field}.fits.bz2`
2. Open your terminal and execute the `rsync` command. *Note: Adjust the destination path `/your/raw/fits/path/` to wherever you have enough hard drive space to store the raw compressed files.*

```bash
rsync -avz --files-from=download_rsync.txt rsync://data.sdss.org/dr17/ /your/raw/fits/path/
```
*Tip: `rsync` is resumable. If your connection drops, simply run the exact same command again. It will verify existing files and only download what is missing.*

### Step 3: Extract Image Cutouts (Batch Processing)
Once the raw FITS files are downloaded, we need to extract the 40x40 pixel cutouts for each galaxy, apply the WCS coordinates, and stack the `u, g, r, i, z` bands.

Run the parallel processing script:
```bash
python download_and_process_data.py
```
**What this does:**
*   Reads the coordinates from the CSV files.
*   Locates the downloaded FITS files.
*   Extracts the 40x40x5 data cubes.
*   Saves them in chunks (e.g., `X_part_1.npy`, `y_part_1.npy`, `ids_part_1.npy`) into the `data/extracted_data/interim_batches/` folder. 
*   *Note: This script uses multiprocessing. It automatically tracks progress and can be safely restarted if interrupted.*

### Step 4: Merge Interim Batches
Once all galaxies have been processed into the `interim_batches` folder, they must be concatenated into the final train and test arrays.

Run the merging script:
```bash
python merge_final.py
```
**What this does:**
*   Loads all `_part_X.npy` files from the interim folder.
*   Uses the `ids_part_X.npy` files to cross-reference with `train.csv` and `test.csv`.
*   Strictly separates the data to prevent data leakage.
*   Saves the final, ready-to-train arrays into `data/extracted_data/final/`.

## 4. Final Verification
After completing Step 4, verify that your `data/extracted_data/final/` directory contains the following files:

*   `X_train.npy` (Shape should be roughly `69571, 40, 40, 5`)
*   `y_train.npy` (Shape should be roughly `69571, 4`)
*   `X_test.npy`  (Shape should be roughly `29817, 40, 40, 5`)
*   `y_test.npy`  (Shape should be roughly `29817, 4`)

You are now ready to run the model training scripts (e.g., `denseNet_1.py`, `pasquet_inception_1.py`).