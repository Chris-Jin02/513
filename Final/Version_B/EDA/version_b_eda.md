# Version B EDA Summary

## Purpose

This EDA checks whether the shared filtered data is ready for collaborative-filtering experiments. It focuses on user-item matrix size, sparsity, positive-feedback density, and temporal train/test consistency.

The reproducible notebook is:

- `Final/Version_B/EDA/version_b_cf_eda.ipynb`

The notebook reads from:

- `Final/Data/Pure_Data`

and writes the summary table to:

- `Final/Version_B/Results/version_b_eda_summary.csv`

## Data Snapshot

The current Version B EDA uses the filtered collaborative view:

- `533,018` filtered interactions
- `507,043` train interactions
- `10,069` test interactions
- `16,973` filtered users
- `39,844` filtered recipes
- `14,879` train users
- `38,636` train recipes
- `9,547` positive-feedback test interactions where `rating >= 4`

## Main Findings

### 1. The filtered split is collaborative-filtering ready

All `10,069` test users and all `5,973` test recipes also appear in the train split, so the filtered evaluation set avoids pure cold-start users and recipes. This makes it appropriate for user-kNN, item-kNN, and SVD collaborative filtering.

### 2. The matrix is still extremely sparse

The filtered user-item matrix has `676,272,212` possible user-recipe cells and only `533,018` observed interactions.

- Matrix density: `0.000788`
- Matrix sparsity: `0.999212`

This is the main reason Version B includes both neighborhood methods and SVD, and why Version C later tests hybrid fallback behavior.

### 3. Positive ratings dominate the filtered data

The positive-feedback ratio is high across the split:

- filtered positive ratio: `0.9542`
- train positive ratio: `0.9547`
- test positive ratio: `0.9482`

Because most ratings are positive, Version B treats `rating >= 4` as implicit positive feedback and evaluates Top-N ranking quality instead of only rating prediction.

### 4. Popularity skew remains important

Collaborative filtering can exploit repeated user-item patterns, but recipe popularity is uneven. This makes B0 a necessary baseline and helps explain why popularity can be competitive even when personalized models are more interesting.

## Reproducibility

To regenerate the EDA summary, run:

- `Final/Version_B/EDA/version_b_cf_eda.ipynb`

The notebook assumes the shared data pipeline has already produced the current files under `Final/Data/Pure_Data`.

## Recommended Report Takeaways

- the filtered split is valid for collaborative-filtering evaluation because test users and recipes are present in train
- the user-item matrix is still extremely sparse, so model choice and fallback behavior matter
- high positive-rating skew makes Top-N ranking metrics more useful than raw rating prediction
- SVD is a strong final Version B model, but coverage tradeoffs should be compared against item-kNN and the hybrid Version C models
