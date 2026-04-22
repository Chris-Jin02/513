# Version A EDA Summary

## Purpose

This folder contains the exploratory data analysis materials for the NutriMatch project. The EDA is designed to support the revised proposal directly, especially the parts about:

- the need for a hybrid recommender instead of a single collaborative model
- the importance of low-history users and low-history recipes
- the use of temporal holdout evaluation instead of only random splits
- the practical relevance of recipe metadata such as preparation time, tags, and nutrition

The main reproducible notebook for Version A is:

- `Final/Version_A/EDA/version_a_recipe_eda.ipynb`

The notebook reads from:

- `Final/Data/Pure_Data`

and saves figures into:

- `Final/Version_A/EDA/Figures`

## Data Snapshot

The current EDA is based on the finalized data pipeline outputs:

- `230,543` cleaned recipes
- `1,067,281` cleaned interactions
- `533,018` globally support-filtered interactions
- `16,973` CF-eligible users in the filtered view
- `39,844` CF-eligible recipes in the filtered view
- `55,547` users eligible for temporal holdout evaluation
- `10,069` filtered test interactions available for collaborative-filtering evaluation

The preprocessing pipeline also reports that `4,239` interactions were dropped because their recipe ids did not survive recipe cleaning. This alignment step is important because it keeps the interaction tables consistent with the recipe metadata tables used for content-based and hybrid recommendation.

## Main Findings

### 1. The recommendation problem is highly sparse

The clean user-item matrix is extremely sparse, which strongly supports the use of a hybrid recommendation system. Pure collaborative filtering would ignore a large portion of the recipe catalog and a large share of users with very short histories.

### 2. Low-history users and recipes are central, not edge cases

The cleaned data shows a long-tailed structure:

- `139,921` users have exactly one interaction
- `33,614` users fall into the `low_history` bucket
- `174,775` recipes have only `1-4` interactions
- only `50,755` recipes have enough clean interaction history to clear the current support threshold

This is the clearest empirical reason to keep content-based features and hybrid fallback logic in scope.

### 3. Temporal evaluation is more realistic than a simple random split

The EDA now includes the temporal train/test split summary. This makes the project evaluation setup more realistic and also shows that the support-filtered collaborative test set is much smaller than the raw temporal holdout set. That gap is useful to document because it explains why cold-start and low-history analysis should not rely only on collaborative-filtering metrics.

### 4. Recipe metadata is rich enough for strong content features

The recipe tables contain useful content signals:

- ingredient lists
- informative tags
- preparation time
- nutrition-related fields
- free-text description and combined text features

This supports the proposal's plan to use TF-IDF, metadata similarity, and practical rule-based filters such as maximum cooking time or dietary exclusions.

### 5. Ranking metrics are more appropriate than only rating prediction

The rating distribution is heavily skewed toward high scores, which makes ranking-focused evaluation more natural for this project. This aligns with the proposal's emphasis on Top-N recommendation and metrics such as Precision@K, Recall@K, and NDCG@K.

## Figures

The current report-ready figures are:

- `recipe_metadata_distributions.png`
- `recipe_submission_trend.png`
- `rating_distribution.png`
- `interaction_density_distributions.png`
- `history_bucket_overview.png`
- `temporal_split_summary.png`
- `top_informative_tags.png`
- `correlation_heatmap.png`

These figures are intended to be reused in the final report and presentation.

## Reproducibility

To regenerate the EDA figures and notebook outputs, run the notebook at:

- `Final/Version_A/EDA/version_a_recipe_eda.ipynb`

The notebook assumes the data pipeline has already been run and that the current files exist under `Final/Data/Pure_Data`.

## Recommended Report Takeaways

The most reusable report-level conclusions from this EDA are:

- the dataset is large enough to support meaningful recommender experiments, but sparse enough that pure collaborative filtering is not sufficient
- low-history users and recipes are common, so cold-start handling is a core requirement rather than a stretch feature
- the temporal evaluation setup reveals a substantial gap between the full cleaned interaction space and the subset that remains evaluable for collaborative filtering
- recipe metadata is rich and interpretable, which makes content-based recommendation and explanation-friendly hybrid ranking both practical
