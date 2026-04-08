from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Paths
DATA_DIR = Path("Final/Data/Pure_Data")
RECIPE_FILE = DATA_DIR / "recipe_model_table.csv"
INTERACTION_FILE = DATA_DIR / "interactions_filtered.csv"

# 2. Load data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    recipes = pd.read_csv(RECIPE_FILE)
    interactions = pd.read_csv(INTERACTION_FILE)
    return recipes, interactions


# 3. Helper functions
def safe_parse_json_list(value) -> list[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [text]

# 4. Popularity Baseline
def build_popularity_baseline(
    recipes: pd.DataFrame,
    interactions: pd.DataFrame,
    min_ratings: int = 10
) -> pd.DataFrame:
    """
    Build popularity baseline using:
    - rating count
    - average rating
    - weighted popularity score
    """

    recipe_stats = (
        interactions.groupby("recipe_id")["rating"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "rating_count", "mean": "avg_rating"})
    )

    recipe_stats = recipe_stats[recipe_stats["rating_count"] >= min_ratings].copy()

    # Simple weighted popularity score
    recipe_stats["popularity_score"] = (
        recipe_stats["avg_rating"] * np.log1p(recipe_stats["rating_count"])
    )

    popularity_df = recipe_stats.merge(
        recipes[["id", "name", "minutes", "rating_count", "rating_mean"]],
        left_on="recipe_id",
        right_on="id",
        how="left"
    )

    popularity_df = popularity_df.sort_values(
        by=["popularity_score", "avg_rating", "rating_count_x"],
        ascending=False
    ).reset_index(drop=True)

    return popularity_df


def get_top_popular_recipes(popularity_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    columns_to_show = [
        "recipe_id",
        "name",
        "avg_rating",
        "rating_count_x",
        "popularity_score",
        "minutes",
    ]
    existing_cols = [c for c in columns_to_show if c in popularity_df.columns]
    return popularity_df[existing_cols].head(top_n)


# 5. Content-Based Recommendation
def prepare_content_features(recipes: pd.DataFrame) -> pd.DataFrame:
    df = recipes.copy()

    # Ensure text columns exist
    for col in ["name", "description", "tags_text", "ingredients_text", "combined_text"]:
        if col not in df.columns:
            df[col] = ""

    df["name"] = df["name"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["tags_text"] = df["tags_text"].fillna("").astype(str)
    df["ingredients_text"] = df["ingredients_text"].fillna("").astype(str)

    # If combined_text not already built, rebuild it
    if "combined_text" not in df.columns or df["combined_text"].isna().all():
        df["combined_text"] = (
            df["name"] + " " +
            df["description"] + " " +
            df["tags_text"] + " " +
            df["ingredients_text"]
        ).str.strip()

    # Add quick recipe hint as text feature
    if "quick_recipe" in df.columns:
        df["quick_text"] = df["quick_recipe"].apply(
            lambda x: "quick_recipe" if bool(x) else ""
        )
    else:
        df["quick_text"] = ""

    df["content_text_final"] = (
        df["combined_text"].fillna("").astype(str) + " " +
        df["quick_text"].fillna("").astype(str)
    ).str.lower().str.strip()

    return df


def build_content_model(recipes: pd.DataFrame):
    df = prepare_content_features(recipes).reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    tfidf_matrix = vectorizer.fit_transform(df["content_text_final"])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    id_to_index = pd.Series(df.index, index=df["id"]).to_dict()

    return df, vectorizer, tfidf_matrix, similarity_matrix, id_to_index


def recommend_similar_recipes(
    recipe_id: int,
    recipes_processed: pd.DataFrame,
    similarity_matrix: np.ndarray,
    id_to_index: dict[int, int],
    top_n: int = 10
) -> pd.DataFrame:
    if recipe_id not in id_to_index:
        raise ValueError(f"Recipe ID {recipe_id} not found.")

    idx = id_to_index[recipe_id]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # remove itself
    sim_scores = [x for x in sim_scores if x[0] != idx][:top_n]

    rec_indices = [i for i, _ in sim_scores]
    rec_scores = [s for _, s in sim_scores]

    recs = recipes_processed.iloc[rec_indices][
        ["id", "name", "minutes", "rating_mean", "rating_count", "quick_recipe"]
    ].copy()

    recs["similarity_score"] = rec_scores
    return recs.reset_index(drop=True)


# 6. User-level Content-Based Recommendation
def recommend_for_user_content_based(
    user_id: int,
    interactions: pd.DataFrame,
    recipes_processed: pd.DataFrame,
    similarity_matrix: np.ndarray,
    id_to_index: dict[int, int],
    user_positive_threshold: float = 4.0,
    top_seed_recipes: int = 5,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Recommend recipes for a user based on recipes they rated highly.
    """

    user_interactions = interactions[interactions["user_id"] == user_id].copy()
    if user_interactions.empty:
        raise ValueError(f"User {user_id} not found.")

    liked = user_interactions[user_interactions["rating"] >= user_positive_threshold].copy()

    if liked.empty:
        liked = user_interactions.sort_values(by="rating", ascending=False).head(top_seed_recipes)
    else:
        liked = liked.sort_values(by="rating", ascending=False).head(top_seed_recipes)

    seen_recipe_ids = set(user_interactions["recipe_id"].tolist())

    score_dict: dict[int, list[float]] = {}

    for rid in liked["recipe_id"]:
        if rid not in id_to_index:
            continue

        idx = id_to_index[rid]
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        for rec_idx, score in sim_scores:
            rec_id = int(recipes_processed.iloc[rec_idx]["id"])

            if rec_id == rid or rec_id in seen_recipe_ids:
                continue

            if rec_id not in score_dict:
                score_dict[rec_id] = []
            score_dict[rec_id].append(score)

    rows = []
    for rec_id, scores in score_dict.items():
        rows.append({
            "recipe_id": rec_id,
            "content_score": float(np.mean(scores))
        })

    recs = pd.DataFrame(rows)
    if recs.empty:
        return recs

    recs = recs.merge(
        recipes_processed[["id", "name", "minutes", "rating_mean", "rating_count", "quick_recipe"]],
        left_on="recipe_id",
        right_on="id",
        how="left"
    )

    recs = recs.sort_values(by="content_score", ascending=False).head(top_n).reset_index(drop=True)
    return recs

# 7. Save outputs
def save_outputs(
    popularity_top10: pd.DataFrame,
    similar_sample: pd.DataFrame,
    user_recs: pd.DataFrame | None = None
) -> None:
    output_dir = DATA_DIR / "model_a_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    popularity_top10.to_csv(output_dir / "top10_popularity_baseline.csv", index=False)
    similar_sample.to_csv(output_dir / "sample_content_based_recommendations.csv", index=False)

    if user_recs is not None and not user_recs.empty:
        user_recs.to_csv(output_dir / "sample_user_content_recommendations.csv", index=False)


# 8. Main
def main() -> None:
    print("Loading data...")
    recipes, interactions = load_data()

    print("Building popularity baseline...")
    popularity_df = build_popularity_baseline(recipes, interactions, min_ratings=10)
    top10_popular = get_top_popular_recipes(popularity_df, top_n=10)

    print("\nTop 10 Popular Recipes:")
    print(top10_popular)

    print("\nBuilding content-based model...")
    recipes_processed, vectorizer, tfidf_matrix, sim_matrix, id_to_index = build_content_model(recipes)

    # sample recipe-based recommendations
    sample_recipe_id = int(recipes_processed["id"].iloc[0])
    seed_recipe = recipes_processed.loc[recipes_processed["id"] == sample_recipe_id, ["id", "name"]]

    print("\nSeed Recipe:")
    print(seed_recipe)

    similar_sample = recommend_similar_recipes(
        recipe_id=sample_recipe_id,
        recipes_processed=recipes_processed,
        similarity_matrix=sim_matrix,
        id_to_index=id_to_index,
        top_n=10
    )

    print("\nTop Similar Recipes:")
    print(similar_sample)

    # sample user-based content recommendations
    sample_user_id = int(interactions["user_id"].iloc[0])
    print(f"\nBuilding user-level content-based recommendations for user {sample_user_id}...")

    try:
        user_recs = recommend_for_user_content_based(
            user_id=sample_user_id,
            interactions=interactions,
            recipes_processed=recipes_processed,
            similarity_matrix=sim_matrix,
            id_to_index=id_to_index,
            user_positive_threshold=4.0,
            top_seed_recipes=5,
            top_n=10
        )
        print("\nUser-level Content-Based Recommendations:")
        print(user_recs)
    except ValueError as e:
        print(f"Could not build user recommendations: {e}")
        user_recs = None

    print("\nSaving outputs...")
    save_outputs(top10_popular, similar_sample, user_recs)

    print("\nDone.")
    print(f"Saved files to: {DATA_DIR / 'model_a_outputs'}")


if __name__ == "__main__":
    main()
