# Data Folder

This folder contains the reproducible Food.com data pipeline for the final project.

## What is tracked in Git

- `Data_Code/foodcom_data_pipeline.py`
- `Data_Code/foodcom_data_pipeline.ipynb`
- `Pure_Data/preprocessing_summary.json`

## What is not tracked in Git

Large raw and processed CSV files are intentionally ignored because they exceed normal GitHub file size limits.

## How to regenerate the data

Open `Data_Code/foodcom_data_pipeline.ipynb` and run all cells once.

The notebook will:

1. Download raw data into `Raw_Data/`
2. Clean and preprocess the dataset
3. Save ready-to-use outputs into `Pure_Data/`
