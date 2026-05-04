# Version B EDA Plan

## Objective
This EDA focuses on collaborative-filtering readiness for Version B.

## Scope
- Analyze collaborative-filtering data from `Final/Data/Pure_Data`.
- Confirm whether the data supports stable Top-N recommendation experiments.

## Planned Checks
1. User count, recipe count, and interaction count.
2. Rating distribution.
3. Positive interaction ratio where `rating >= 4`.
4. User interaction distribution (long-tail behavior).
5. Recipe interaction distribution (popularity skew).
6. User-item matrix sparsity.
7. Train/test split consistency and sanity checks.

## Expected EDA Outputs
- Tables and figures describing CF data characteristics.
- A short summary of potential risks (sparsity, cold users/items, skew).
- Inputs for model and evaluation decisions in Version B.
