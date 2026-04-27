# Version A Result Summary

## Run Context

Final Version A evaluation used the filtered temporal split:

- Train file: `Final/Data/Pure_Data/interactions_train_filtered.csv`
- Test file: `Final/Data/Pure_Data/interactions_test_filtered.csv`
- Evaluated users: `10,069`
- Primary cutoff: `K = 10`
- Final configuration: `fast_dev_mode = False`, `run_full_filtered_eval = True`, `run_svd_model = True`
- Evaluation errors: `0`

The main comparison table is stored in `version_a_metrics.csv`. The duplicate full filtered check in `version_a_full_filtered_metrics.csv` confirms the same A0/A3/A4/A5 metrics without saving recommendation detail.

## Final Model Choice

The recommended final Version A model is **A4 Content plus Popularity Reranker**.

A0 has the best offline exact-match score, but it is a non-personalized popularity model with extremely low catalog coverage. A4 is the better final Version A model because it keeps the recommendation logic metadata-first, personalized, interpretable, and broad-coverage while slightly improving ranking quality over A3.

## Metric Comparison at K=10

| Model | Precision@10 | Recall@10 | NDCG@10 | Hit@10 | Coverage@10 | Seconds/User | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A0 Bayesian popularity | 0.001003 | 0.010031 | 0.004618 | 0.010031 | 0.000621 | 0.00349 | Strongest exact-match baseline, but recommends almost the same popular recipes to everyone |
| A2 TF-IDF item-to-item | 0.000218 | 0.002185 | 0.001134 | 0.002185 | 0.448520 | 0.28565 | Interpretable content baseline, but slowest model |
| A3 User-profile TF-IDF | 0.000268 | 0.002681 | 0.001321 | 0.002681 | 0.457397 | 0.05787 | Personalized metadata model with strong coverage |
| A4 Content plus popularity | 0.000268 | 0.002681 | 0.001344 | 0.002681 | 0.452997 | 0.06133 | Best Version A final model: same hit rate as A3, slightly better NDCG, broad coverage |
| A5 SVD semantic content | 0.000209 | 0.002086 | 0.000956 | 0.002086 | 0.595067 | 0.01562 | Fast semantic comparison model with highest coverage but weaker exact-match accuracy |

## Tuning Findings

- **A0 popularity smoothing:** `m = 100` performed best among the tested smoothing constants.
- **A3 TF-IDF profile:** the best final TF-IDF setup used `max_features = 30000`, `min_df = 3`, and `ngram_range = (1, 2)`.
- **A4 reranking weights:** the best reranker used `content_weight = 0.70`, `popularity_weight = 0.25`, and `practical_weight = 0.05`.
- **A5 semantic components:** `128` SVD components outperformed `64` components and was selected as the final A5 variant.

## Runtime Findings

The full notebook run took about `8,318` seconds, or `138.6` minutes.

The main runtime bottleneck was A2 item-to-item evaluation, which took about `2,876` seconds by itself. A5 was much faster than A2 because it scores dense semantic embeddings instead of repeatedly searching sparse item profiles. A4 adds a modest runtime cost over A3, but the cost is acceptable for a final metadata-first model.

## How to Present the Result

The strongest presentation story is:

1. A0 proves that popularity is a very strong baseline for Food.com exact-match evaluation.
2. A0's weakness is diversity: `Coverage@10 = 0.000621`.
3. A3 and A4 trade exact-match accuracy for personalization, interpretability, and broad catalog coverage.
4. A4 is the final Version A model because it adds quality control and practical preference signals while preserving A3's hit rate and improving NDCG.
5. A5 shows that semantic compression increases coverage and speed, but literal TF-IDF user profiles were more accurate for this dataset.

## Limitations

The Top-N exact-match metrics are modest because the test split usually gives each user one held-out recipe from a large candidate catalog. A recommendation can be reasonable and metadata-similar while still missing the single hidden recipe. For this reason, Version A should be evaluated as a lightweight, interpretable, broad-coverage recommendation track rather than as the highest possible exact-match recommender.

## Report Assets

Useful figures for the report or presentation:

- `Figures/version_a_model_metrics_at_10.png`
- `Figures/version_a_quality_vs_runtime_at_10.png`
- `Figures/version_a_a4_rerank_weight_tuning.png`
- `Figures/version_a_a4_rerank_weight_heatmap.png`
- `Figures/version_a_content_explanation_terms.png`
- `Figures/version_a_sample_recommendation_scores.png`
- `Figures/version_a_hit_rank_distribution_at_10.png`
