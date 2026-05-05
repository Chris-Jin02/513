# Version C Workflow

## Model direction
Hybrid recommendation: combine CF (SVD) and content (TF-IDF) signals.

## Steps

1. Run `Experiments/version_c_full_experiment.ipynb` with `DEBUG_MODE=True` first
2. Verify all 5 models (C0-C4) produce valid metrics
3. Set `DEBUG_MODE=False`, `FULL_RUN=True` and run full evaluation
4. Review cross-version comparison table
5. Write up findings in Results

## Shared inputs
- `Final/Data/Pure_Data`

## Important
- Do not redo raw data cleaning here
- Keep evaluation protocol identical to Version A and B
- Use same positive threshold (rating >= 4) and same temporal split
