# Version A Results

This folder should hold generated metrics, recommendation examples, and report-ready summaries for Version A.

## Expected files

- `version_a_metrics.csv`
- `version_a_runtime.csv`
- `version_a_example_recommendations.csv`
- `version_a_model_notes.md`

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

Recommended columns for `version_a_runtime.csv`:

- `model_id`
- `fit_seconds`
- `recommend_seconds_total`
- `recommend_seconds_per_user`
- `evaluated_users`
- `candidate_count`

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
