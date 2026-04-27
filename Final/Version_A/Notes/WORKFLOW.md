# Version A Workflow

Version A is the lightweight metadata-first recommendation track.

## Recommended order

1. Keep the existing EDA as the evidence base.
2. Follow `GOOGLE_DRIVE_COLAB_RUN_GUIDE.md` if running through Colab web.
3. Run the Google Drive mount cell in `Experiments/version_a_full_experiment.ipynb`.
4. Run the GPU verification cell and confirm that Colab exposes a GPU.
5. Run the notebook in sampled mode.
6. Check the progress bars, metric plots, tuning plots, runtime plots, and recommendation examples.
7. Increase the evaluation sample size or enable full filtered evaluation.
8. Save metric tables, recommendation examples, figures, and short conclusions in `Results/`.

## Model priority

Build in this order:

| Priority | Model | Reason |
| --- | --- | --- |
| 1 | A0 Bayesian popularity | Fast baseline and sanity check |
| 2 | A2 TF-IDF item similarity | Core content model and demo-friendly examples |
| 3 | A3 user-profile content | Personalized recommendation without heavy training |
| 4 | A4 content plus popularity rerank | Best candidate for Version A final result |
| 5 | A5 SVD semantic content | Required semantic content comparison model |

## Completed deliverables

- Main notebook: `Experiments/version_a_full_experiment.ipynb`
- Final result summary: `Results/version_a_model_notes.md`
- Final metric table: `Results/version_a_metrics.csv`
- Full filtered confirmation table: `Results/version_a_full_filtered_metrics.csv`
- Runtime and tuning tables: `Results/version_a_phase_runtime.csv`, `Results/version_a_tuning_results.csv`
- Report figures: `Results/Figures/*.png`
- Model artifacts: `Experiments/model_artifacts/`

Final selected model:

- A4 Content plus popularity reranker

## Evaluation rule

Primary comparison should use the same temporal setup as the other versions:

- train from `interactions_train_filtered.csv` when comparing directly with CF models
- test on `interactions_test_filtered.csv`
- report Precision@10, Recall@10, NDCG@10, catalog coverage, and runtime

Secondary analysis may use `interactions_train.csv` and `interactions_test.csv` to show that content-based methods cover more sparse users and recipes than collaborative filtering.

## Runtime guardrails

- Do not build a full recipe-by-recipe similarity matrix.
- Use sparse TF-IDF operations and retrieve only top candidates.
- Start evaluation on a sampled user set, then scale to the filtered test users.
- Keep TF-IDF feature limits moderate, for example `20,000` to `30,000` features.
- Keep `TruncatedSVD` enabled, but tune only a small number of component counts.
- Keep the notebook in sampled mode until A0, A2, A3, A4, and A5 all produce valid metric tables.

## Model artifacts

The notebook saves trained model artifacts under:

- `Final/Version_A/Experiments/model_artifacts/`

Expected artifacts include:

- A0 Bayesian popularity score vector and metadata
- TF-IDF vectorizer, sparse matrix, and candidate arrays
- A4 rerank weights
- A5 fitted SVD model, SVD components, explained variance, and item embeddings
- a `model_artifact_manifest.json` file listing generated artifacts

## Important

- Do not redo raw data cleaning here.
- Use `Final/Data/Pure_Data` as the canonical input.
- Use only train interactions to compute popularity scores during evaluation.
- Keep this version distinct from `Version_B` collaborative filtering and `Version_C` final hybrid work.
