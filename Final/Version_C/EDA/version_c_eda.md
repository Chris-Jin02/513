# Version C EDA Plan

## Objective
Analyze user history distribution to justify the hybrid approach.

## Key questions
1. How many users have sparse history (< 10 interactions)?
2. How many users have rich history (>= 20 interactions)?
3. Does CF performance correlate with user history count?
4. Does content recommendation help more for sparse users?

## Data source
- `Final/Data/Pure_Data/interactions_train_filtered.csv`
- `Final/Data/Pure_Data/recipe_model_table.csv`

## Expected outputs
- User history count distribution plot
- Sparsity analysis supporting the hybrid strategy
