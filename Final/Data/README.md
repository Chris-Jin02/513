# Data Folder

This folder contains the reproducible Food.com data pipeline for the final project.

## Role in the current project framework

`Final/Data` is the shared technical base for the whole project.
All model versions should read from `Pure_Data/`, and no one should duplicate the full preprocessing workflow inside `Version_A`, `Version_B`, or `Version_C`.

## Small files that are safe to track in Git

- `Data_Code/foodcom_data_pipeline.py`
- `Data_Code/foodcom_data_pipeline.ipynb`
- `Data_Code/model_a_popularity_content.py`
- `Pure_Data/preprocessing_summary.json`
- `Pure_Data/temporal_split_summary.json`

## Large files that should stay out of Git

Large raw and processed CSV files are intentionally ignored because they exceed normal GitHub file size limits.

## Recommended regeneration path

Run the Python pipeline script from the workspace root:

```bash
python Final/Data/Data_Code/foodcom_data_pipeline.py
```

Use a Python environment that has at least `pandas` and `numpy` installed.

The notebook is still useful for inspection, but the script is the source of truth for reproducible outputs.

## What the pipeline produces

The pipeline will:

1. Download raw data into `Raw_Data/` if the files are not already present
2. Clean recipes and interactions, then align interactions to cleaned recipe ids
3. Save ready-to-use outputs into `Pure_Data/`

Key outputs in `Pure_Data/`:

- `recipes_clean.csv`: cleaned recipe metadata for all valid recipes
- `interactions_clean.csv`: cleaned interaction table after deduplication, rating validation, and recipe-id alignment
- `interactions_filtered.csv`: global support-filtered interaction table for collaborative-filtering analysis
- `interactions_train.csv`: temporal training split built before support filtering
- `interactions_test.csv`: per-user temporal holdout set
- `interactions_train_filtered.csv`: support-filtered training split for collaborative-filtering evaluation
- `interactions_test_filtered.csv`: test rows that remain evaluable after applying train-side support filtering
- `recipe_model_table.csv`: full recipe metadata table with both clean-history and filtered-history statistics
- `user_statistics.csv`: user-level history summary with clean and filtered support counts
- `preprocessing_summary.json`: high-level data cleaning summary
- `temporal_split_summary.json`: evaluation split summary

## Which file to use for which task

- Use `recipes_clean.csv` for EDA on recipe metadata.
- Use `interactions_clean.csv` when analyzing the full interaction distribution or studying sparse users/items.
- Use `interactions_filtered.csv` for global collaborative-filtering exploration on the fully cleaned dataset.
- Use `interactions_train_filtered.csv` and `interactions_test_filtered.csv` for collaborative-filtering model evaluation.
- Use `recipe_model_table.csv` for content-based and hybrid models because it preserves the full recipe universe while exposing both clean and filtered history fields.

In the current framework, each version folder should use these shared outputs for its own EDA, experiments, and results.

## Important modeling note

`recipe_model_table.csv` now keeps two different interaction views:

- `clean_*` columns describe all cleaned interactions
- `filtered_*` columns describe the support-filtered collaborative subset

This is important for the final project because low-history recipes should not be mistaken for true zero-history recipes.

The preprocessing summary also reports how many cleaned interaction rows were dropped because their recipe ids did not survive recipe cleaning.
