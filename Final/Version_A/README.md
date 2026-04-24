# Version A

## Target outcome

Version A is now the **lightweight metadata-first recommendation track**.

The goal is to build several strong, fast recommenders that can run on a normal laptop and still give the final project a convincing comparison story. This version should be stronger than a plain popularity baseline, but it should not become a heavy collaborative-filtering or deep-learning track.

## Updated task

Version A owns:

- popularity baselines
- TF-IDF content-based recommendation
- user-profile content recommendation
- a lightweight content-plus-popularity reranker
- practical filters such as maximum time, quick recipes, and ingredient or tag constraints

Version A does not own:

- full collaborative filtering
- matrix factorization on the user-item matrix
- the final cross-version hybrid with collaborative signals
- LLM explanation features

Those should remain separate from this version so the three project tracks stay distinguishable.

## Model lineup

| ID | Model | Why it belongs in Version A | Expected cost |
| --- | --- | --- | --- |
| A0 | Bayesian popularity baseline | Stronger and fairer than raw average rating because it controls for low rating counts | Very low |
| A1 | Time/tag constrained popularity | Useful demo model for quick meals and dietary constraints | Very low |
| A2 | TF-IDF item-to-item content model | Uses recipe name, description, tags, and ingredients; strong for similar-recipe search | Low |
| A3 | User-profile content model | Builds a user preference vector from liked recipes; gives personalized results without collaborative training | Low to medium |
| A4 | Content plus popularity reranker | Combines A3 relevance with train-only popularity and practical filters; likely Version A's best final model | Medium |

Optional stretch only if runtime is comfortable:

- latent semantic content model using `TruncatedSVD` on TF-IDF features

## Folder roles

- `EDA/`: completed Version A exploratory analysis and report-ready figures
- `Experiments/`: implementation plan, experiment log, and future model/evaluation scripts
- `Results/`: metric tables, recommendation examples, and report summaries
- `Notes/`: task definition, model roadmap, evaluation protocol, and limitations

## Shared inputs

All experiments should read from `Final/Data/Pure_Data`.

Recommended files:

- `recipe_model_table.csv`
- `interactions_train.csv`
- `interactions_test.csv`
- `interactions_train_filtered.csv`
- `interactions_test_filtered.csv`
- `preprocessing_summary.json`
- `temporal_split_summary.json`

Use train interactions for all popularity statistics during evaluation. Do not use full-history rating columns as scoring features when measuring test performance.

## Success criteria

Version A should produce:

- at least three evaluated models: A0, A2, and A4
- Top-N metrics such as Precision@10, Recall@10, NDCG@10, and coverage
- a small table comparing runtime and quality
- several readable recommendation examples for the final report or presentation
- a clear explanation of why metadata helps sparse users and low-history recipes
