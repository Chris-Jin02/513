# Version A Experiments

This folder will contain the implementation and experiment logs for the lightweight metadata-first recommendation track.

## Planned implementation files

Recommended future files:

- `version_a_recommenders.py`: model classes and scoring helpers
- `run_version_a_experiments.py`: train/evaluate script for A0 through A4
- `version_a_metrics.py`: Precision@K, Recall@K, NDCG@K, coverage, and runtime helpers
- `experiment_log.md`: short record of what was run and what changed

Keep the code small. The first complete script should evaluate A0, A2, A3, and A4 before adding optional extras.

## Model order

1. A0 Bayesian popularity
2. A2 TF-IDF item-to-item content
3. A3 user-profile content
4. A4 content plus popularity reranker
5. Optional A5 latent semantic content model

## Output location

Save generated outputs under:

- `Final/Version_A/Results/`

Expected outputs:

- `version_a_metrics.csv`
- `version_a_runtime.csv`
- `version_a_example_recommendations.csv`
- `version_a_model_notes.md`

## Development rule

Use the narrowest reliable check first:

1. run on a few users
2. run on a fixed sample of around `1,000` users
3. run on the full filtered temporal test set

Do not optimize before the basic ranking and metric pipeline is correct.
