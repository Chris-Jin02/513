# Version B

Purpose:
- this folder is reserved for one teammate's full recommendation version

Suggested use:
- use this folder for a distinct model family from `Version_A`
- a good default is collaborative filtering, such as kNN or a lightweight factorization method

Workflow inside this folder:
1. `EDA/`: this version's own EDA using shared processed data
2. `Experiments/`: model building, fitting, validation, and tuning
3. `Results/`: outputs scored on the shared temporal split
4. `Notes/`: decisions, limitations, and summary write-ups

Rules:
- do not duplicate raw data or full preprocessing here
- use `Final/Data/Pure_Data` as the canonical shared input
- keep this version clearly distinguishable from `Version_A` and `Version_C`
