# CS 513 Final Project Proposal

## Title
**NutriMatch: A Hybrid Recipe Recommendation System with Dietary Constraints and Explainable Recommendations**

## 1. Project Summary
We propose to build a practical **recommendation system** that suggests recipes based on a user's historical preferences, ingredient interests, dietary constraints, and preparation-time needs. The project is centered on a direct comparison between popularity-based, content-based, collaborative-filtering, and hybrid recommendation approaches.

This proposal is stronger than a purely conceptual plan because the data stage is already underway. We have already completed dataset acquisition, cleaning, feature construction, and collaborative-filtering support filtering, so the remaining work can focus on modeling, evaluation, and demonstration quality.

Our goal is to produce a project that is:

- clearly aligned with the course requirement for a recommendation system
- methodologically strong because it compares multiple recommenders under the same evaluation protocol
- interesting and distinctive because it uses food, nutrition, and recipe metadata instead of a standard movie dataset
- realistic for a three-person team on normal laptops

## 2. Problem Statement and Research Questions
Recipe discovery is harder than generic search because users often care about several constraints at once: taste, ingredients, time, nutrition, and dietary restrictions. A recipe that is popular overall may still be a poor recommendation for a user who needs fast meals, avoids certain ingredients, or prefers a specific cuisine-related tag profile.

The main question of this project is:

**How can a lightweight hybrid recommender combine user interaction history and recipe metadata to produce accurate, interpretable, and practically useful recipe suggestions?**

We will study three more specific questions:

- How much does collaborative filtering improve over a simple popularity baseline?
- How much do metadata features help in sparse-user or sparse-item settings?
- Can explainable dietary and time filters improve practical usefulness without making the system too complex?

## 3. Why This Is a Strong CS 513 Project
This topic fits the project rubric very well:

- **Clear recommendation focus**: the problem is explicitly a user-item recommendation task
- **Method comparison**: the project naturally supports multiple baselines and direct evaluation
- **Richer data than standard classroom examples**: Food.com provides ratings plus ingredients, tags, nutrition, descriptions, and timestamps
- **Practical relevance**: recipe recommendation is easy to motivate and easy to demo live
- **Controlled ambition**: the core system is fully feasible on CPU, while the optional explanation layer adds originality without becoming a dependency

## 4. Dataset and Current Progress

### Primary Dataset
We use the public **[Food.com Recipe & Review Data](https://berd-platform.de/records/0455w-b0633)** dataset, with the complete download available via its linked **[Kaggle page](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)**.

The raw dataset includes tables such as `RAW_recipes.csv` and `RAW_interactions.csv`, which together provide:

- user-recipe ratings
- recipe titles and descriptions
- ingredient lists
- recipe tags and categories
- preparation time
- nutrition-related metadata
- cooking steps
- interaction timestamps

This is a strong fit for our project because it supports both:

- **collaborative filtering**, through explicit user-item ratings
- **content-based recommendation**, through ingredients, tags, descriptions, nutrition, and time features

### Data Work Already Completed
We have already built a reproducible preprocessing pipeline for this dataset. The current outputs are:

- `recipes_clean.csv`: `230,543` cleaned recipes
- `interactions_clean.csv`: `1,067,281` cleaned user-recipe interactions
- `interactions_filtered.csv`: `533,018` interactions retained for collaborative filtering
- `recipe_model_table.csv`: a recipe-level modeling table with metadata plus interaction statistics

Before cleaning, the raw data contained:

- `231,637` recipe rows
- `1,132,367` interaction rows

Our pipeline currently performs the following core steps:

- removes duplicate or invalid records
- keeps the most recent interaction for repeated user-recipe pairs
- aligns interactions to recipe ids that survive recipe cleaning
- parses tags, ingredients, steps, and nutrition into model-friendly features
- expands recipe metadata into text and numeric fields
- filters users and recipes to a minimum of `5` ratings each for collaborative-filtering experiments

After support filtering, the collaborative-filtering subset contains:

- `16,973` users
- `39,844` recipes
- `533,018` interactions

### Why This Dataset Works Well
This dataset is especially valuable for our project because:

- it includes explicit ratings rather than only clicks or views
- it contains rich item metadata needed for explainable and content-based recommendations
- it includes timestamps, which support realistic train/test splitting
- it supports practical filters such as quick recipes, nutrition constraints, and ingredient exclusions
- it is large enough to be interesting, but still manageable with sparse matrices and lightweight models

### Two Modeling Views of the Data
One important design choice in our project is to keep two related but different views of the dataset:

- a **filtered interaction view** for collaborative filtering, where users and recipes have enough support to train reliable similarity or factorization models
- a **full recipe metadata view** for content-based and hybrid recommendation, where recipes can still be represented even if they have little or no interaction history

This distinction is important because it lets us study cold-start and sparsity more honestly instead of forcing one preprocessing choice onto every model.

### Backup Dataset
If we need a simpler fallback for any reason, our backup dataset will be **[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)**. However, because the Food.com pipeline is already built and the data is already cleaned, dataset access risk is now much lower than it was at the planning stage.

## 5. Proposed Recommendation Methods
We will implement and compare the following methods.

### Baseline 1: Popularity-Based Ranking
We will recommend globally popular recipes using a weighted score based on average rating and rating count. This baseline is easy to interpret and gives us a minimal benchmark for non-personalized ranking.

### Baseline 2: Content-Based Recommendation
We will build recipe profiles from available metadata such as:

- ingredients
- tags
- description text
- preparation time
- nutrition-related fields

Our current cleaned tables already include tokenized fields such as ingredient text, tag text, and combined text, so this method is well supported by the existing data pipeline. A likely implementation is TF-IDF plus cosine similarity, with optional rule-based filters for time or dietary constraints.

This model is especially useful for:

- sparse users with limited history
- sparse or new recipes with weak interaction support
- explainable recommendations based on shared ingredients or tags

### Model 3: Collaborative Filtering
We will train collaborative-filtering recommenders on the filtered user-item interaction matrix. Candidate methods include:

- user-based kNN
- item-based kNN
- matrix factorization such as SVD, if runtime remains manageable

This is the main personalization model in settings where sufficient rating history exists.

### Model 4: Hybrid Recommendation System
Our intended final system is a hybrid recommender that combines collaborative and content-based signals. The hybrid model will likely use either:

- a weighted combination of normalized collaborative and content scores
- a fallback strategy that increases content reliance when user or item history is sparse

This design is attractive because it balances:

- personalization
- interpretability
- cold-start handling
- manageable computation cost

We also plan to support practical user-side filters such as maximum preparation time and dietary exclusions. These filters are especially appropriate for recipe recommendation because they improve usefulness in ways that standard media recommenders usually cannot.

### Optional Stretch Extension: Agentic Explanation Layer
As an optional extension, we may add a lightweight explanation layer that:

- converts natural-language preferences into structured filters
- explains why a recipe was recommended
- suggests simple substitutions for dietary restrictions

Example input:

> "Recommend quick high-protein dinners with no peanuts and no pork."

This extension would not require training a large model. If included, it would use an LLM only for preference parsing and explanation generation, while keeping the recommendation logic itself lightweight and reproducible.

## 6. Experimental Design and Evaluation
Because the course emphasizes comparison of methods, evaluation will be a central part of the project.

### Primary Task Definition
Our primary task is **Top-N recommendation**, not only rating prediction. The main goal is to rank useful recipes near the top of the recommendation list for each user.

### Train/Test Protocol
To reduce leakage and better reflect real usage, we will use a **per-user temporal holdout** strategy based on interaction dates:

- earlier interactions will be used for training
- each user's later interaction(s) will be held out for evaluation
- the same split protocol will be used across all comparable models

This is more realistic than a fully random row split because it respects the order in which user preferences are observed.

### Offline Evaluation Metrics
For recommendation quality, we plan to report:

- Precision@K
- Recall@K
- NDCG@K
- coverage

If time allows, we may also report:

- diversity
- novelty

If we include a rating-prediction model such as SVD, we may additionally report:

- RMSE
- MAE

However, these will be treated as **secondary metrics**, not the main project objective.

### Comparison Questions
We want our experiments to answer the following:

- Does collaborative filtering outperform popularity-based recommendation?
- Does the hybrid system outperform either single-method system on ranking metrics?
- Does content-based recommendation help in sparse-user or sparse-item settings?
- Do time and dietary filters improve practical usefulness while preserving recommendation quality?

### Cold-Start and Practical-Use Evaluation
We specifically want to evaluate two scenarios that matter in this domain:

- **sparse-history users**, where collaborative filtering has limited signal
- **low-history recipes**, where metadata must carry more of the recommendation quality

Because our recipe modeling table preserves metadata even for recipes outside the collaborative subset, we can study these cases directly instead of ignoring them.

### Qualitative Evaluation
In addition to offline metrics, we will show example recommendation cases to demonstrate:

- personalized results
- interpretable reasoning based on ingredients, tags, or time
- practical usefulness for realistic dietary requests

## 7. Software and Tools
Planned tools:

- Python
- pandas
- numpy
- scikit-learn
- Surprise or another lightweight recommendation library
- matplotlib / seaborn
- Streamlit or notebook-based demo
- optional LLM API for the explanation layer

All of these are feasible on standard laptops. No heavy deep learning training is required.

## 8. Deliverables
Our final submission will include:

- a cleaned and documented data pipeline
- version-specific EDA and experiment materials built on the shared data pipeline
- popularity-based, content-based, and collaborative-filtering baselines
- a final hybrid recommender
- comparison tables and evaluation figures
- a short live demo or interactive notebook
- final presentation slides
- a final report with references and documentation

## 9. Team Work Division
To keep the workload balanced for a three-person group, we will use one shared data foundation plus three parallel model versions.

### Shared Foundation
- maintain and document the completed data pipeline
- keep the processed train/test artifacts fixed for fair comparison

### Member A: Version A
- build a complete model line using the shared data
- maintain version-specific EDA, experiments, and results
- likely focus on popularity and content-based recommendation

### Member B: Version B
- build a complete model line using the shared data
- maintain version-specific EDA, experiments, and results
- likely focus on collaborative filtering

### Member C: Version C
- build a complete model line using the shared data
- maintain version-specific EDA, experiments, and results
- likely focus on hybrid recommendation or another distinct model family

### Shared Responsibilities
- keep evaluation rules aligned across versions
- interpret model results together
- decide the final system story
- write the final report and polish the presentation

## 10. Workload and Compute Control
This project is designed to look ambitious while remaining technically manageable.

We will control scope by:

- avoiding large neural network training
- using CPU-friendly models first
- building the project in layers: popularity -> content -> collaborative -> hybrid
- keeping the LLM-based explanation layer optional
- reusing the completed data pipeline instead of revisiting preprocessing unnecessarily

This keeps the project realistic for a three-person team while still leaving room for a polished final demo.

## 11. Timeline

### Completed
- finalized the dataset choice
- built the reproducible data cleaning pipeline
- generated cleaned recipe and interaction tables
- applied collaborative-filtering support thresholds
- completed initial version-level EDA materials

### Next Phase
- each member creates version-specific EDA inside their own version folder
- each member builds the first end-to-end experiment for their chosen model line

### Following Phase
- improve and tune each version's experiments
- compare results across the three version tracks under the same temporal split

### Final Phase
- finalize the best version or hybrid system
- run final comparisons and create result tables
- prepare the demo, slides, and final paper
- optionally add the explanation layer if time permits

## 12. Risks and Mitigation

### Risk 1: Interaction sparsity limits collaborative filtering
Mitigation:

- use support thresholds for the collaborative subset
- rely on content-based metadata features when history is weak
- use a hybrid fallback strategy rather than one single model

### Risk 2: Metadata is noisy or inconsistent
Mitigation:

- use cleaned tokenized text fields instead of raw string columns
- compare multiple feature subsets rather than depending on one metadata source
- keep qualitative inspection in the workflow so bad features are easier to detect

### Risk 3: Scope grows too large because of the explanation feature
Mitigation:

- treat the explanation layer as optional
- keep the recommender itself independent from any LLM dependency
- prioritize evaluation quality and model comparison first

### Risk 4: Team workload becomes uneven
Mitigation:

- assign clear ownership by modeling stage
- keep shared checkpoints for evaluation and reporting
- integrate the final results through a common comparison pipeline

## 13. Expected Outcome
We expect the hybrid system to achieve the best overall balance of ranking quality, coverage, and practical usefulness. More specifically, we expect:

- popularity to provide a weak but interpretable baseline
- collaborative filtering to perform well for users and recipes with sufficient history
- content-based recommendation to be especially valuable for sparse cases and interpretability
- the hybrid model to deliver the strongest overall system because it combines the strengths of both approaches

## 14. Conclusion
This project is a strong fit for CS 513 because it combines a real recommendation problem, a rich dataset, multiple comparable methods, and a realistic path to an engaging demo. The proposal is also lower risk than a typical early-stage plan because the dataset pipeline has already been completed and the remaining work is focused on modeling, evaluation, and presentation.

If approved, this project should produce a final system that is both technically solid and easy to present: a good match for a high-quality final project in recommendation systems.
