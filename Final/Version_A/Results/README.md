# Version A Results

This folder should hold generated metrics, recommendation examples, and report-ready summaries for Version A.

## Final selection

The final Version A model is **A4 Content plus popularity reranker**.

Use `version_a_model_notes.md` as the report-facing result summary. It explains why A4 is preferred over A0 even though A0 has the strongest exact-match score.

## Expected files

- `version_a_metrics.csv`
- `version_a_per_user_metrics.csv`
- `version_a_tuning_results.csv`
- `version_a_example_recommendations.csv`
- `version_a_phase_runtime.csv`
- `version_a_config.json`
- `version_a_model_notes.md`
- `Figures/*.png`

Model artifacts are saved outside `Results/`, next to the notebook:

- `Final/Version_A/Experiments/model_artifacts/`

## Metric table schema

Recommended columns for `version_a_metrics.csv`:

- `model_id`
- `model_name`
- `split`
- `k`
- `evaluated_users`
- `precision_at_k`
- `recall_at_k`
- `ndcg_at_k`
- `catalog_coverage_at_k`

## Runtime table schema

Recommended columns for `version_a_phase_runtime.csv`:

- `phase`
- `seconds`
- optional phase metadata such as users, errors, features, and nonzero TF-IDF entries

Runtime columns are also included in `version_a_metrics.csv` for each evaluated model.

## Final K=10 results

| Model | Precision@10 | Recall@10 | NDCG@10 | Hit@10 | Coverage@10 | Seconds/User |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 Bayesian popularity | 0.001003 | 0.010031 | 0.004618 | 0.010031 | 0.000621 | 0.00349 |
| A2 TF-IDF item-to-item | 0.000218 | 0.002185 | 0.001134 | 0.002185 | 0.448520 | 0.28565 |
| A3 User-profile TF-IDF | 0.000268 | 0.002681 | 0.001321 | 0.002681 | 0.457397 | 0.05787 |
| A4 Content plus popularity | 0.000268 | 0.002681 | 0.001344 | 0.002681 | 0.452997 | 0.06133 |
| A5 SVD semantic content | 0.000209 | 0.002086 | 0.000956 | 0.002086 | 0.595067 | 0.01562 |

## Tuning table schema

Recommended columns for `version_a_tuning_results.csv`:

- `tuning_family`
- `model_id`
- `k`
- `precision_at_k`
- `recall_at_k`
- `ndcg_at_k`
- tuned parameter columns such as `tuned_m`, `max_features`, `min_df`, `ngram_range`, and rerank weights

## Example recommendations

Recommended columns for `version_a_example_recommendations.csv`:

- `model_id`
- `user_id`
- `rank`
- `recipe_id`
- `recipe_name`
- `score`
- `minutes`
- `matched_tags_or_ingredients`
- `reason`

## Reporting focus

The final Version A write-up should emphasize:

- A4 as the strongest metadata-first model if metrics support it
- A0 as a fair baseline
- A2 and A3 as interpretable content models
- coverage and sparse-history usefulness as Version A's major advantage
- tuning and runtime plots as evidence that the final model was chosen deliberately without making training too heavy
- A5 as the required semantic content comparison against the literal TF-IDF models
