# Version B Workflow

Version B is the collaborative-filtering recommendation track.

## Recommended order

1. Run `EDA/version_b_cf_eda.ipynb` to confirm the filtered split is CF-ready.
2. Review `EDA/version_b_eda.md` and `Results/version_b_eda_summary.csv`.
3. Run `Experiments/version_b_full_experiment.ipynb`.
4. Verify B0 through B4 all produce metric rows.
5. Review tuning results for user-kNN, item-kNN, and SVD.
6. Save metric tables, recommendation examples, runtime tables, and figures in `Results/`.
7. Use `Results/version_b_model_notes.md` for the final write-up.

## Model priority

Build and evaluate in this order:

| Priority | Model | Reason |
| --- | --- | --- |
| 1 | B0 Bayesian popularity | Fast baseline and sanity check |
| 2 | B1 user-based kNN CF | Direct collaborative personalization baseline |
| 3 | B2 item-based kNN CF implicit | Stronger catalog coverage from item co-occurrence |
| 4 | B3 item-based kNN CF rating-weighted | Tests whether centered ratings improve similarity |
| 5 | B4 SVD collaborative filtering | Best Version B quality model and reusable CF component |

## Completed deliverables

- EDA notebook: `EDA/version_b_cf_eda.ipynb`
- EDA summary: `EDA/version_b_eda.md`
- Main notebook: `Experiments/version_b_full_experiment.ipynb`
- Final result summary: `Results/version_b_model_notes.md`
- Final metric table: `Results/version_b_metrics.csv`
- Tuning table: `Results/version_b_tuning_results.csv`
- Runtime table: `Results/version_b_phase_runtime.csv`
- Per-user metrics: `Results/version_b_per_user_metrics.csv`
- Recommendation outputs: `Results/version_b_example_recommendations.csv`, `Results/version_b_top10_recommendations.csv`
- Report figures: `Results/Figures/*.png`

Final selected model:

- B4 SVD collaborative filtering with 64 components

## Evaluation rule

Primary comparison should use the same filtered temporal setup as the other versions:

- train from `interactions_train_filtered.csv`
- test on `interactions_test_filtered.csv`
- treat `rating >= 4` as positive feedback
- report Precision@10, Recall@10, NDCG@10, Hit@10, catalog coverage, and runtime

All user-user similarities, item-item similarities, popularity scores, and SVD factors must be fitted from train interactions only.

## Runtime guardrails

- Keep kNN neighbor grids small enough to run on a normal laptop.
- Use sparse user-item matrices for collaborative filtering.
- Avoid materializing dense full user-item score matrices.
- Start with a sampled/debug run if changing the notebook, then run the full filtered evaluation.
- Keep B4 component tuning focused on a few values such as `32`, `64`, and `128`.

## Important

- Do not redo raw data cleaning here.
- Use `Final/Data/Pure_Data` as the canonical input.
- Keep Version B distinct from Version A content-based work and Version C hybrid work.
- Document the main CF tradeoff clearly: B4 has the best exact-match quality, while item-kNN models have much broader catalog coverage.
