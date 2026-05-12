# Version B

## Target outcome

Version B is the **collaborative-filtering recommendation track**.

The goal is to test user-item interaction models on the shared filtered temporal split and provide a clear comparison point against Version A's metadata-first models and Version C's hybrid models.

## Updated task

Version B owns:

- Bayesian popularity baseline for comparison
- user-based kNN collaborative filtering
- item-based kNN collaborative filtering
- rating-weighted item kNN collaborative filtering
- latent-factor collaborative filtering with `TruncatedSVD`

Version B does not own:

- TF-IDF content recommendation
- metadata-first filtering and reranking
- final cross-version hybrid ranking
- LLM explanation features

Those should remain separate so the three project tracks stay easy to compare.

## Model lineup

| ID | Model | Why it belongs in Version B | Expected cost |
| --- | --- | --- | --- |
| B0 | Bayesian popularity baseline | Establishes a non-personalized reference point | Very low |
| B1 | User-based kNN CF | Uses similar users' histories for personalized ranking | Low |
| B2 | Item-based kNN CF implicit | Uses co-interaction signals from positive feedback | Medium |
| B3 | Item-based kNN CF rating-weighted | Tests whether centered rating weights improve item similarity | Medium |
| B4 | SVD collaborative filtering | Learns latent user and item factors from the user-item matrix | Medium |

## Folder roles

- `EDA/`: collaborative-filtering readiness analysis and EDA notebook
- `Experiments/`: main Version B experiment notebook
- `Results/`: metric tables, tuning results, recommendation examples, and figures
- `Notes/`: workflow and decision records

## Main experiment notebook

Run the full Version B workflow from:

- `Experiments/version_b_full_experiment.ipynb`

The notebook contains:

- data loading for the shared filtered temporal split
- B0 through B4 model implementations
- hyperparameter tuning for kNN neighbors and SVD components
- Top-N evaluation at K=10
- metric, runtime, tuning, per-user, and recommendation export into `Results/`
- result figures exported into `Results/Figures/`

The EDA notebook is:

- `EDA/version_b_cf_eda.ipynb`

## Final result

The completed Version B run evaluated `9,547` positive-feedback test users on the filtered temporal split.

Final model recommendation:

- **B4 SVD collaborative filtering with 64 components**

Why B4 is the final Version B model:

- It has the strongest Precision@10, Recall@10, NDCG@10, and Hit@10 among the Version B models.
- It is much more personalized than the popularity baseline.
- It provides the collaborative-filtering component later reused by Version C.
- Its main tradeoff is lower catalog coverage than item-kNN models, which is useful to discuss in the final comparison.

Primary K=10 result summary:

| Model | Precision@10 | Recall@10 | NDCG@10 | Hit@10 | Coverage@10 | Seconds/User |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 Bayesian popularity | 0.001027 | 0.010265 | 0.004708 | 0.010265 | 0.000621 | 0.000005 |
| B1 User-based kNN CF | 0.001927 | 0.019273 | 0.009841 | 0.019273 | 0.153820 | 0.000614 |
| B2 Item-based kNN implicit | 0.000880 | 0.008799 | 0.004432 | 0.008799 | 0.467724 | 0.003000 |
| B3 Item-based kNN rating-weighted | 0.000681 | 0.006808 | 0.003874 | 0.006808 | 0.544958 | 0.001521 |
| B4 SVD CF 64 | 0.002524 | 0.025244 | 0.013089 | 0.025244 | 0.032638 | 0.000896 |

Detailed interpretation is in:

- `Results/version_b_model_notes.md`

## Shared inputs

All experiments should read from `Final/Data/Pure_Data`.

Recommended files:

- `interactions_filtered.csv`
- `interactions_train_filtered.csv`
- `interactions_test_filtered.csv`
- `recipe_model_table.csv`

Use train interactions for all collaborative statistics during evaluation. Do not use test interactions when building similarity matrices or latent factors.

## Success criteria

Version B should produce:

- at least five evaluated models: B0, B1, B2, B3, and B4
- Top-N metrics such as Precision@10, Recall@10, NDCG@10, hit rate, and coverage
- tuning results for kNN and SVD settings
- runtime comparison across the collaborative models
- recommendation examples for the final report or presentation
- a clear explanation of collaborative filtering strengths and sparsity limitations
