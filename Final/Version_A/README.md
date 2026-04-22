# Version A

Purpose:
- this is your version folder
- use it for your full recommendation line from model-specific EDA to evaluation notes

Suggested use:
- use this folder for the simplest strong baseline or the first chosen model family
- a good default is popularity plus content-based recommendation

Workflow inside this folder:
1. `EDA/`: your own EDA for this version using shared processed data
2. `Experiments/`: model building, fitting, validation, and tuning
3. `Results/`: outputs scored on the shared temporal split
4. `Notes/`: decisions, limitations, and summary write-ups

Rules:
- do not duplicate raw data or full preprocessing here
- use `Final/Data/Pure_Data` as the canonical shared input
- keep this version clearly distinguishable from `Version_B` and `Version_C`
