# Version A Workflow

Version A is the lightweight metadata-first recommendation track.

## Recommended order

1. Keep the existing EDA as the evidence base.
2. Implement a train-only Bayesian popularity baseline.
3. Implement TF-IDF item-to-item recipe similarity.
4. Implement user-profile content recommendation from liked recipes.
5. Add a lightweight content-plus-popularity reranker.
6. Evaluate the models under the shared temporal split.
7. Save metric tables, recommendation examples, and short conclusions in `Results/`.

## Model priority

Build in this order:

| Priority | Model | Reason |
| --- | --- | --- |
| 1 | A0 Bayesian popularity | Fast baseline and sanity check |
| 2 | A2 TF-IDF item similarity | Core content model and demo-friendly examples |
| 3 | A3 user-profile content | Personalized recommendation without heavy training |
| 4 | A4 content plus popularity rerank | Best candidate for Version A final result |
| 5 | Optional SVD content model | Only if the main models run comfortably |

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
- Treat `TruncatedSVD` as optional, not required.

## Important

- Do not redo raw data cleaning here.
- Use `Final/Data/Pure_Data` as the canonical input.
- Use only train interactions to compute popularity scores during evaluation.
- Keep this version distinct from `Version_B` collaborative filtering and `Version_C` final hybrid work.
