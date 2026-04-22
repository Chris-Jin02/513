# Version B Workflow

This folder is for one teammate's recommendation version.

Recommended order:
1. add this version's own EDA note or notebook in `EDA/`
2. build, fit, validate, and tune the main model in `Experiments/`
3. save metric tables, recommendation examples, and final outputs in `Results/`
4. summarize strengths, weaknesses, and example recommendations in `Notes/`

Recommended model direction:
- collaborative filtering
- kNN or lightweight matrix factorization

Shared inputs:
- `Final/Data/Pure_Data`

Important:
- do not redo raw data cleaning here
- keep this version comparable to `Version_A` and `Version_C`
