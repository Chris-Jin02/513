# Version A Experiments

This folder will contain the implementation and experiment logs for the lightweight metadata-first recommendation track.

## Main notebook

Use this notebook for the complete Version A experiment:

- `version_a_full_experiment.ipynb`

It implements Google Drive mounting, GPU runtime verification, the model lineup, tuning loops, progress bars, metric plots, runtime plots, recommendation examples, model artifact exports, and result exports. Training/evaluation is not pre-run in the repository; run the notebook in Colab after confirming that `Final/Data/Pure_Data` exists in Google Drive.

For the Colab web + Google Drive workflow, see:

- `../Notes/GOOGLE_DRIVE_COLAB_RUN_GUIDE.md`

## Planned implementation files

Recommended future files:

- `version_a_recommenders.py`: model classes and scoring helpers
- `run_version_a_experiments.py`: train/evaluate script for A0 through A4
- `version_a_metrics.py`: Precision@K, Recall@K, NDCG@K, coverage, and runtime helpers
- `experiment_log.md`: short record of what was run and what changed

Keep the code small. The first complete script should evaluate A0, A2, A3, A4, and A5 before adding any extra experiments.

The notebook currently keeps these pieces together to make the project easier to run and present. If the code grows too large, split the helper classes into the planned `.py` files later.

## Model order

1. A0 Bayesian popularity
2. A2 TF-IDF item-to-item content
3. A3 user-profile content
4. A4 content plus popularity reranker
5. A5 latent semantic content model

## Output location

Save generated outputs under:

- `Final/Version_A/Results/`

Expected outputs:

- `version_a_metrics.csv`
- `version_a_per_user_metrics.csv`
- `version_a_tuning_results.csv`
- `version_a_runtime.csv`
- `version_a_example_recommendations.csv`
- `version_a_phase_runtime.csv`
- `version_a_config.json`
- `version_a_model_notes.md`
- `Figures/*.png`
- `model_artifacts/*`

## Model artifact folder

The notebook writes trained artifacts into:

- `Final/Version_A/Experiments/model_artifacts/`

This folder is intentionally next to the notebook so Drive inspection and cleanup are straightforward.

## Development rule

Use the narrowest reliable check first:

1. run on a few users
2. run on a fixed sample of around `1,000` users
3. run on the full filtered temporal test set

Do not optimize before the basic ranking and metric pipeline is correct.
