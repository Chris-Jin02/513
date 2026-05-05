# Version C

## Target outcome

Version C is the **hybrid recommendation track**.

The goal is to combine collaborative filtering (from Version B) and content-based recommendation (from Version A) into hybrid models, then show whether the combination improves over either method alone.

## Model lineup

| ID | Model | Description | Expected cost |
| --- | --- | --- | --- |
| C0 | SVD CF baseline | Same SVD method as B4, provides the CF component | Medium |
| C1 | TF-IDF content baseline | Same user-profile method as A3, provides the content component | Low to medium |
| C2 | Weighted hybrid | Normalized CF + content scores with tunable alpha weight | Medium |
| C3 | Switching hybrid | Uses CF for rich-history users, content for sparse users | Medium |
| C4 | RRF hybrid | Reciprocal Rank Fusion of CF and content ranked lists | Medium |

Key comparisons:

- C2 tests whether a simple weighted combination beats either single method
- C3 tests whether adapting to user history density helps
- C4 tests whether rank-based fusion (score-scale independent) is more robust

## Folder roles

- `EDA/`: hybrid-specific analysis (sparsity, user history distribution)
- `Experiments/`: main experiment notebook
- `Results/`: metric tables, figures, recommendation examples
- `Notes/`: workflow and decision records

## Main experiment notebook

Run the full Version C workflow from:

- `Experiments/version_c_full_experiment.ipynb`

The notebook contains:

- data loading for the shared temporal split
- C0 SVD CF baseline with component tuning
- C1 TF-IDF content baseline with max_features tuning
- C2 weighted hybrid with alpha tuning
- C3 switching hybrid with threshold tuning
- C4 RRF hybrid with k tuning
- metric visualizations
- cross-version comparison (A vs B vs C)
- CSV and figure export into `Results/`

The notebook defaults to `DEBUG_MODE=True` for quick testing. For the final run, set `DEBUG_MODE=False` and `FULL_RUN=True`.

## Shared inputs

All experiments read from `Final/Data/Pure_Data`:

- `interactions_filtered.csv`
- `interactions_train_filtered.csv`
- `interactions_test_filtered.csv`
- `recipe_model_table.csv`

## Success criteria

Version C should produce:

- at least five evaluated models: C0, C1, C2, C3, C4
- Top-N metrics: Precision@10, Recall@10, NDCG@10, coverage
- cross-version comparison table (best of A vs best of B vs best of C)
- evidence of whether hybrid methods improve over single methods
- analysis of sparse-user vs rich-user performance differences
