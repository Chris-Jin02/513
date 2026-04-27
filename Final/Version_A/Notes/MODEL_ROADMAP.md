# Version A Model Roadmap

## Final direction

Version A should become a strong **metadata-first recommender** rather than only a simple baseline. The project should still be lightweight: no deep learning, no full collaborative-filtering matrix factorization, and no long training jobs.

The strongest practical route is:

1. build a reliable popularity baseline
2. build content similarity from recipe metadata
3. personalize content results using each user's liked recipes
4. rerank personalized content results with train-only popularity and simple constraints

This creates a good high-score story: Version A is interpretable, fast, sparse-user friendly, and easy to demo.

## Chosen models

### A0 Bayesian popularity

Purpose:

- establish a stronger baseline than raw average rating
- avoid overvaluing recipes with only a few high ratings
- provide a stable fallback for users without enough history

Suggested score:

```text
bayesian_score = (v / (v + m)) * R + (m / (v + m)) * C
```

Where:

- `R` is the recipe's train average rating
- `v` is the recipe's train rating count
- `C` is the global train average rating
- `m` is a minimum-support smoothing constant

### A1 Constrained popularity

Purpose:

- support demo queries such as quick recipes, max cooking time, and ingredient exclusions
- show practical usefulness beyond pure metrics

This model uses the A0 score after filtering candidates.

### A2 TF-IDF item-to-item content

Purpose:

- recommend recipes similar to a seed recipe
- support cold-start items because it relies on metadata rather than interaction history
- generate easy-to-explain examples based on shared tags and ingredients

Recommended text fields:

- recipe name
- description
- tags text
- ingredients text
- optional quick-recipe token

Implementation note:

- fit one `TfidfVectorizer`
- do top-k sparse cosine search per seed recipe
- do not materialize the full item-item similarity matrix

### A3 User-profile content

Purpose:

- produce personalized recommendations without collaborative filtering
- make Version A stronger than item-only similarity

Method:

- collect each user's liked recipes from the training split
- use ratings `>= 4` as positive feedback
- build a user vector as the weighted average of liked recipe TF-IDF vectors
- score candidate recipes by cosine similarity to the user vector
- exclude recipes already seen in training

This should work especially well for users whose liked recipes share ingredients, tags, or cooking styles.

### A4 Content plus popularity reranker

Purpose:

- produce the best Version A final model
- combine personalization, quality control, and practical constraints

Suggested score:

```text
final_score =
  0.70 * normalized_content_score
  + 0.25 * normalized_bayesian_popularity
  + 0.05 * practical_filter_bonus
```

The exact weights should be tuned lightly. Avoid a large grid search. A small set such as `(0.8, 0.2)`, `(0.7, 0.25)`, and `(0.6, 0.35)` is enough.

### A5 Latent semantic content model

Use `TruncatedSVD` on TF-IDF features as a required semantic content comparison model.

Why it may help:

- reduces sparse text features into broader semantic dimensions
- may improve recommendations when exact ingredient/tag overlap is too literal

Runtime control:

- keep the component grid small, such as `64` and `128`
- run sampled evaluation first
- use A5 as a comparison point against A3's literal TF-IDF user-profile model

Saved artifacts:

- fitted `TruncatedSVD` object
- SVD components
- explained variance ratio
- dense item embeddings

## Evaluation plan

Primary metrics:

- Precision@10
- Recall@10
- NDCG@10
- catalog coverage
- average recommendation runtime per user

Primary split:

- train: `interactions_train_filtered.csv`
- test: `interactions_test_filtered.csv`

Secondary sparse-coverage analysis:

- train: `interactions_train.csv`
- test: `interactions_test.csv`

The secondary analysis is useful because content models can recommend across a wider recipe universe than collaborative-filtering models.

## Final result table

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage | Runtime | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| A0 Bayesian popularity | 0.001003 | 0.010031 | 0.004618 | 0.000621 | 0.00349 s/user | Strong exact-match baseline, but low diversity |
| A2 TF-IDF item content | 0.000218 | 0.002185 | 0.001134 | 0.448520 | 0.28565 s/user | Interpretable but slow item-similarity content model |
| A3 User-profile content | 0.000268 | 0.002681 | 0.001321 | 0.457397 | 0.05787 s/user | Personalized metadata model with high coverage |
| A4 Content plus popularity rerank | 0.000268 | 0.002681 | 0.001344 | 0.452997 | 0.06133 s/user | Final Version A model |
| A5 SVD semantic content | 0.000209 | 0.002086 | 0.000956 | 0.595067 | 0.01562 s/user | Fast semantic comparison with highest coverage |

Final interpretation:

- A0 wins on exact-match metrics but is not the best final Version A model because it is non-personalized and has very low catalog coverage.
- A4 is the final Version A model because it preserves A3's hit rate, improves NDCG@10, and maintains broad metadata-driven coverage.
- A5 is useful as a required semantic comparison, but it did not beat A3/A4 on exact-match ranking quality.

## Risks and controls

- If full evaluation is slow, evaluate on a reproducible sample first.
- If TF-IDF is too memory-heavy, lower `max_features`.
- If content scores recommend obscure low-quality recipes, increase the popularity weight in A4.
- If popularity dominates too much, lower its weight and report improved personalization.
- If metrics are modest, emphasize coverage, interpretability, and sparse-history usefulness.
