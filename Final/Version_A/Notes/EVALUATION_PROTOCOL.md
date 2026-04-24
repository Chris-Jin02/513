# Version A Evaluation Protocol

## Task

Evaluate Top-N recommendation. For each user, recommend recipes from the training-time candidate catalog, exclude recipes already seen in training, and check whether the user's held-out recipe appears near the top.

## Primary comparison split

Use this split when comparing with Version B and Version C:

- `Final/Data/Pure_Data/interactions_train_filtered.csv`
- `Final/Data/Pure_Data/interactions_test_filtered.csv`

This keeps Version A comparable to collaborative-filtering models that require train-side user and recipe support.

## Secondary coverage split

Use this split for Version A's own sparse-user and sparse-recipe story:

- `Final/Data/Pure_Data/interactions_train.csv`
- `Final/Data/Pure_Data/interactions_test.csv`

This split shows where metadata methods have a practical advantage over pure collaborative filtering.

## Metrics

Report:

- `Precision@10`
- `Recall@10`
- `NDCG@10`
- `catalog_coverage@10`
- evaluated user count
- average runtime per evaluated user

Optional:

- `Precision@5`
- `Recall@5`
- `NDCG@5`
- examples grouped by user history bucket

## Leakage rules

- Popularity scores must be computed from train interactions only.
- User profiles must be built from train interactions only.
- The held-out test recipe must not be used to build the user profile.
- Recipe metadata can be used because item metadata is available before recommendation.
- Full-history rating columns in `recipe_model_table.csv` may be displayed but should not be scoring features in evaluation.

## Candidate rules

Default candidate set:

- recipes appearing in the training split for direct comparison

Secondary candidate set:

- all recipes in `recipe_model_table.csv` with valid metadata

Always exclude:

- recipes the user already rated in training
- recipes filtered out by explicit practical constraints

## Practical runtime rule

Start with a fixed random sample of test users, for example `1,000` users. After the code is correct, run the full filtered test set.

For the final report, clearly label whether metrics came from:

- sampled filtered evaluation
- full filtered evaluation
- full clean temporal evaluation
