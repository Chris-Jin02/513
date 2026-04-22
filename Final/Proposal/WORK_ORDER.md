# Work Order and Team Workflow

## Core idea
The project now uses a shared-foundation plus parallel-version workflow.

Shared foundation:
- `Data` is the canonical dataset pipeline and output source.
- `Proposal` is the common planning reference.

Parallel version stage:
- each team member builds one complete recommendation version
- all versions use the same processed data
- the versions differ mainly by model choice, not by data source

This means the project is no longer organized as "one person only does data, another person only does evaluation." Instead, after the shared foundation is finished, each member owns a full modeling path.

## Recommended work order

### Stage 1: Shared foundation
Completed or nearly completed:
- dataset download and cleaning
- filtered collaborative subset creation
- recipe and user modeling tables
- temporal split artifacts

At this stage, `Final/Data` becomes the common base for everyone else.
Ongoing EDA work belongs inside each version folder rather than in one central EDA location.

### Stage 2: Each member creates one model version
Each version should follow the same internal order:
1. do that version's own EDA using the shared processed data
2. implement, train, validate, and tune the chosen recommender as one experiment workflow
3. save metrics, example outputs, and observations in that version's results area
4. record strengths, weaknesses, and next-step ideas in notes

Important:
- members should not redo the full data pipeline
- EDA is now part of each member's personal version workflow, not a separate shared task
- model building, training, validation, and optimization should stay together inside each version's experiment workflow

## Suggested version split

### Version A
Suggested direction:
- popularity baseline
- content-based recommendation

Main purpose:
- provide the strongest non-collaborative baseline
- handle recipes that still have useful metadata even if interaction history is weak

### Version B
Suggested direction:
- collaborative filtering
- kNN or lightweight matrix factorization

Main purpose:
- test whether interaction-based recommendation improves over the simpler baselines

### Version C
Suggested direction:
- hybrid recommendation
- or another distinct model family if the hybrid should be built later

Main purpose:
- combine the strengths of metadata-driven and interaction-driven recommendation
- improve practical coverage for sparse-history cases

## Shared evaluation rule
All versions should be judged under the same setup:
- same train and test split
- same Top-N recommendation task
- same metrics such as `Precision@K`, `Recall@K`, and `NDCG@K`

This is necessary because the proposal is built around method comparison.

## What each member should produce
Each version folder should contain:
- a version-specific EDA note or notebook
- experiment code that includes modeling, fitting, validation, and tuning
- result outputs and metric summaries
- a short conclusion on where the model works well or poorly

## Final integration stage
After the three versions are finished:
1. compare results across all versions
2. decide whether one version is clearly best or whether a hybrid should be the final system
3. choose strong example cases for the demo
4. write the final report and prepare slides

## Practical recommendation
To avoid duplicated effort:
- keep `Data` centralized
- let each member focus on one distinct model path
- use common evaluation code whenever possible
- keep outputs easy to compare across versions
