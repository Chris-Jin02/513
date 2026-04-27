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
| A5 | Latent semantic content model | Uses `TruncatedSVD` on TF-IDF features to compare semantic content ranking against literal TF-IDF ranking | Medium |

Required semantic comparison:

- latent semantic content model using `TruncatedSVD` on TF-IDF features

## Folder roles

- `EDA/`: completed Version A exploratory analysis and report-ready figures
- `Experiments/`: implementation plan, experiment log, and future model/evaluation scripts
- `Results/`: metric tables, recommendation examples, and report summaries
- `Notes/`: task definition, model roadmap, evaluation protocol, and limitations

## Main experiment notebook

Run the full Version A workflow from:

- `Experiments/version_a_full_experiment.ipynb`

For Colab web + Google Drive execution, follow:

- `Notes/GOOGLE_DRIVE_COLAB_RUN_GUIDE.md`

This notebook contains:

- data loading for the shared temporal split
- Google Drive mounting for Colab
- GPU runtime verification for Colab
- A0 through A4 model implementations
- required A5 latent semantic content model
- tqdm progress bars for fitting and evaluation phases
- hyperparameter tuning for popularity smoothing, TF-IDF settings, and rerank weights
- metric visualizations for Precision@K, Recall@K, NDCG@K, hit rate, coverage, and runtime
- recommendation examples and content-explanation plots
- CSV and figure export into `Results/`
- trained model artifacts exported into `Experiments/model_artifacts/`

The notebook defaults to sampled evaluation so it can be tested quickly. For the final run, increase the sample size or enable full filtered evaluation in the configuration cell.

## Final result

The completed Version A run evaluated `10,069` users on the filtered temporal split with no evaluation errors.

Final model recommendation:

- **A4 Content plus popularity reranker**

Why A4 is the final Version A model:

- A0 has the highest exact-match score, but it is non-personalized and covers only `0.062%` of the candidate catalog at K=10.
- A4 keeps A3's personalized content hit rate while slightly improving NDCG@10.
- A4 reaches `45.30%` catalog coverage at K=10, making it much stronger for the metadata-first story.
- A4 remains lightweight enough for the project scale.

Primary K=10 result summary:

| Model | Precision@10 | Recall@10 | NDCG@10 | Hit@10 | Coverage@10 | Seconds/User |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 Bayesian popularity | 0.001003 | 0.010031 | 0.004618 | 0.010031 | 0.000621 | 0.00349 |
| A2 TF-IDF item-to-item | 0.000218 | 0.002185 | 0.001134 | 0.002185 | 0.448520 | 0.28565 |
| A3 User-profile TF-IDF | 0.000268 | 0.002681 | 0.001321 | 0.002681 | 0.457397 | 0.05787 |
| A4 Content plus popularity | 0.000268 | 0.002681 | 0.001344 | 0.002681 | 0.452997 | 0.06133 |
| A5 SVD semantic content | 0.000209 | 0.002086 | 0.000956 | 0.002086 | 0.595067 | 0.01562 |

Detailed interpretation is in:

- `Results/version_a_model_notes.md`

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

- at least five evaluated models: A0, A2, A3, A4, and A5
- Top-N metrics such as Precision@10, Recall@10, NDCG@10, and coverage
- a small table comparing runtime and quality
- several readable recommendation examples for the final report or presentation
- a clear explanation of why metadata helps sparse users and low-history recipes
