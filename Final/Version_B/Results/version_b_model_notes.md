# Version B Result Summary

## Run Context

Final Version B evaluation used the filtered temporal split:

- Train file: `Final/Data/Pure_Data/interactions_train_filtered.csv`
- Test file: `Final/Data/Pure_Data/interactions_test_filtered.csv`
- Positive feedback rule: `rating >= 4`
- Evaluated positive-feedback users: `9,547`
- Primary cutoff: `K = 10`

The main comparison table is stored in `version_b_metrics.csv`.

## Final Model Choice

The recommended final Version B model is **B4 SVD collaborative filtering with 64 components**.

B4 has the strongest ranking quality across Version B, with the best Precision@10, Recall@10, NDCG@10, and Hit@10. The tradeoff is catalog coverage: item-kNN models recommend a wider slice of the recipe catalog, while B4 concentrates more heavily on latent collaborative patterns.

## Metric Comparison at K=10

| Model | Precision@10 | Recall@10 | NDCG@10 | Hit@10 | Coverage@10 | Seconds/User | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0 Bayesian popularity | 0.001027 | 0.010265 | 0.004708 | 0.010265 | 0.000621 | 0.000005 | Fast non-personalized baseline with extremely low coverage |
| B1 User-based kNN CF | 0.001927 | 0.019273 | 0.009841 | 0.019273 | 0.153820 | 0.000614 | Strong neighborhood baseline with much better coverage than popularity |
| B2 Item-based kNN implicit | 0.000880 | 0.008799 | 0.004432 | 0.008799 | 0.467724 | 0.003000 | Broad catalog coverage but weaker exact-match quality |
| B3 Item-based kNN rating-weighted | 0.000681 | 0.006808 | 0.003874 | 0.006808 | 0.544958 | 0.001521 | Highest Version B coverage, but weakest ranking quality |
| B4 SVD CF 64 | 0.002524 | 0.025244 | 0.013089 | 0.025244 | 0.032638 | 0.000896 | Best Version B final model by ranking quality |

## Tuning Findings

- **B0 popularity smoothing:** `m = 100` performed best among the tested smoothing constants.
- **B1 user-kNN:** `k_neighbors = 50` performed best among the tested user-neighbor values.
- **B2 implicit item-kNN:** `k_neighbors = 100` produced the best B2 result.
- **B3 rating-weighted item-kNN:** `k_neighbors = 10` produced the best B3 result.
- **B4 SVD:** `64` components slightly outperformed `32` and clearly outperformed `128`.

## Runtime Findings

All Version B models are practical on the filtered split. B0 is nearly instant, while the kNN item models are the slowest because they need broader item-neighborhood scoring. B4 is a good final collaborative model because it gives the best quality while staying under `0.001` seconds per evaluated user in the saved run.

## How to Present the Result

The strongest presentation story is:

1. B0 shows that popularity is a hard baseline on this dataset.
2. B1 improves exact-match quality and coverage by using similar users.
3. B2 and B3 show that item-neighborhood models can greatly expand catalog coverage.
4. B4 is the final Version B model because latent collaborative factors produce the best ranking quality.
5. B4's lower coverage motivates Version C's hybrid comparison with content-based signals.

## Limitations

Version B only evaluates users and recipes that survive the filtered collaborative split. That makes the CF comparison fair, but it does not solve the broader cold-start problem. The final report should compare this result against Version A's content coverage and Version C's hybrid behavior.

## Report Assets

Useful figures for the report or presentation:

- `Figures/version_b_knn_tuning_curves.png`
- `Figures/version_b_svd_tuning_curves.png`
- `Figures/version_b_sample_recommendation_scores.png`
- `Figures/version_b_data_sparsity_context.png`
