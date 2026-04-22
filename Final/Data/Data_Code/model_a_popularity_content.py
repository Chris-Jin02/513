from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def infer_workspace_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "Final").exists():
            return candidate
    raise FileNotFoundError("Could not locate the workspace root that contains a Final directory.")


WORKSPACE_ROOT = infer_workspace_root()
DATA_DIR = WORKSPACE_ROOT / "Final" / "Data" / "Pure_Data"
RECIPE_FILE = DATA_DIR / "recipe_model_table.csv"


def resolve_interaction_file() -> Path:
    preferred_file = DATA_DIR / "interactions_train_filtered.csv"
    if preferred_file.exists():
        return preferred_file
    return DATA_DIR / "interactions_filtered.csv"


INTERACTION_FILE = resolve_interaction_file()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    recipes = pd.read_csv(RECIPE_FILE)
    interactions = pd.read_csv(INTERACTION_FILE)
    return recipes, interactions


def build_popularity_baseline(
    recipes: pd.DataFrame,
    interactions: pd.DataFrame,
    min_ratings: int = 10,
) -> pd.DataFrame:
    recipe_stats = (
        interactions.groupby("recipe_id")["rating"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "rating_count", "mean": "avg_rating"})
    )

    recipe_stats = recipe_stats[recipe_stats["rating_count"] >= min_ratings].copy()
    recipe_stats["popularity_score"] = recipe_stats["avg_rating"] * np.log1p(recipe_stats["rating_count"])

    popularity_df = recipe_stats.merge(
        recipes[["id", "name", "minutes", "rating_count", "rating_mean"]],
        left_on="recipe_id",
        right_on="id",
        how="left",
    )

    popularity_df = popularity_df.sort_values(
        by=["popularity_score", "avg_rating", "rating_count_x"],
        ascending=False,
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
    existing_cols = [column for column in columns_to_show if column in popularity_df.columns]
    return popularity_df[existing_cols].head(top_n)


def prepare_content_features(recipes: pd.DataFrame) -> pd.DataFrame:
    df = recipes.copy()

    for column in ["name", "description", "tags_text", "ingredients_text", "combined_text"]:
        if column not in df.columns:
            df[column] = ""

    df["name"] = df["name"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["tags_text"] = df["tags_text"].fillna("").astype(str)
    df["ingredients_text"] = df["ingredients_text"].fillna("").astype(str)

    if "combined_text" not in df.columns or df["combined_text"].isna().all():
        df["combined_text"] = (
            df["name"] + " " + df["description"] + " " + df["tags_text"] + " " + df["ingredients_text"]
        ).str.strip()

    if "quick_recipe" in df.columns:
        df["quick_text"] = df["quick_recipe"].apply(lambda value: "quick_recipe" if bool(value) else "")
    else:
        df["quick_text"] = ""

    df["content_text_final"] = (
        df["combined_text"].fillna("").astype(str) + " " + df["quick_text"].fillna("").astype(str)
    ).str.lower().str.strip()

    return df


def build_content_model(recipes: pd.DataFrame) -> tuple[pd.DataFrame, TfidfVectorizer, Any, dict[int, int]]:
    df = prepare_content_features(recipes).reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
    )

    tfidf_matrix = vectorizer.fit_transform(df["content_text_final"])
    id_to_index = pd.Series(df.index, index=df["id"]).to_dict()

    return df, vectorizer, tfidf_matrix, id_to_index


def top_k_similar_indices(tfidf_matrix, row_index: int, *, top_k: int, exclude_index: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    similarities = linear_kernel(tfidf_matrix[row_index], tfidf_matrix).ravel()

    if exclude_index is not None:
        similarities[exclude_index] = -np.inf

    top_k = max(1, min(top_k, similarities.shape[0] - (1 if exclude_index is not None else 0)))
    candidate_indices = np.argpartition(similarities, -top_k)[-top_k:]
    ordered_indices = candidate_indices[np.argsort(similarities[candidate_indices])[::-1]]
    ordered_scores = similarities[ordered_indices]

    valid_mask = np.isfinite(ordered_scores)
    return ordered_indices[valid_mask], ordered_scores[valid_mask]


def recommend_similar_recipes(
    recipe_id: int,
    recipes_processed: pd.DataFrame,
    tfidf_matrix,
    id_to_index: dict[int, int],
    top_n: int = 10,
) -> pd.DataFrame:
    if recipe_id not in id_to_index:
        raise ValueError(f"Recipe ID {recipe_id} not found.")

    idx = id_to_index[recipe_id]
    rec_indices, rec_scores = top_k_similar_indices(
        tfidf_matrix,
        idx,
        top_k=top_n,
        exclude_index=idx,
    )

    recs = recipes_processed.iloc[rec_indices][
        ["id", "name", "minutes", "rating_mean", "rating_count", "quick_recipe", "history_bucket", "eligible_for_cf"]
    ].copy()
    recs["similarity_score"] = rec_scores

    return recs.reset_index(drop=True)


def recommend_for_user_content_based(
    user_id: int,
    interactions: pd.DataFrame,
    recipes_processed: pd.DataFrame,
    tfidf_matrix,
    id_to_index: dict[int, int],
    user_positive_threshold: float = 4.0,
    top_seed_recipes: int = 5,
    top_n: int = 10,
) -> pd.DataFrame:
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
    candidate_pool_size = max(top_n * 10, 50)

    for recipe_id in liked["recipe_id"]:
        if recipe_id not in id_to_index:
            continue

        recipe_index = id_to_index[recipe_id]
        rec_indices, rec_scores = top_k_similar_indices(
            tfidf_matrix,
            recipe_index,
            top_k=candidate_pool_size,
            exclude_index=recipe_index,
        )

        for rec_index, score in zip(rec_indices, rec_scores):
            rec_id = int(recipes_processed.iloc[rec_index]["id"])

            if rec_id in seen_recipe_ids:
                continue

            if rec_id not in score_dict:
                score_dict[rec_id] = []
            score_dict[rec_id].append(float(score))

    rows = [
        {"recipe_id": recipe_id, "content_score": float(np.mean(scores))}
        for recipe_id, scores in score_dict.items()
    ]
    recs = pd.DataFrame(rows)

    if recs.empty:
        return recs

    recs = recs.merge(
        recipes_processed[
            ["id", "name", "minutes", "rating_mean", "rating_count", "quick_recipe", "history_bucket", "eligible_for_cf"]
        ],
        left_on="recipe_id",
        right_on="id",
        how="left",
    )

    recs = recs.sort_values(by="content_score", ascending=False).head(top_n).reset_index(drop=True)
    return recs


def save_outputs(
    popularity_top10: pd.DataFrame,
    similar_sample: pd.DataFrame,
    user_recs: pd.DataFrame | None = None,
) -> None:
    output_dir = DATA_DIR / "model_a_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    popularity_top10.to_csv(output_dir / "top10_popularity_baseline.csv", index=False)
    similar_sample.to_csv(output_dir / "sample_content_based_recommendations.csv", index=False)

    if user_recs is not None and not user_recs.empty:
        user_recs.to_csv(output_dir / "sample_user_content_recommendations.csv", index=False)


def main() -> None:
    print("Loading data...")
    recipes, interactions = load_data()

    print("Building popularity baseline...")
    popularity_df = build_popularity_baseline(recipes, interactions, min_ratings=10)
    top10_popular = get_top_popular_recipes(popularity_df, top_n=10)

    print("\nTop 10 Popular Recipes:")
    print(top10_popular)

    print("\nBuilding content-based model...")
    recipes_processed, vectorizer, tfidf_matrix, id_to_index = build_content_model(recipes)

    sample_recipe_id = int(recipes_processed["id"].iloc[0])
    seed_recipe = recipes_processed.loc[recipes_processed["id"] == sample_recipe_id, ["id", "name"]]

    print("\nSeed Recipe:")
    print(seed_recipe)

    similar_sample = recommend_similar_recipes(
        recipe_id=sample_recipe_id,
        recipes_processed=recipes_processed,
        tfidf_matrix=tfidf_matrix,
        id_to_index=id_to_index,
        top_n=10,
    )

    print("\nTop Similar Recipes:")
    print(similar_sample)

    sample_user_id = int(interactions["user_id"].iloc[0])
    print(f"\nBuilding user-level content-based recommendations for user {sample_user_id}...")

    try:
        user_recs = recommend_for_user_content_based(
            user_id=sample_user_id,
            interactions=interactions,
            recipes_processed=recipes_processed,
            tfidf_matrix=tfidf_matrix,
            id_to_index=id_to_index,
            user_positive_threshold=4.0,
            top_seed_recipes=5,
            top_n=10,
        )
        print("\nUser-level Content-Based Recommendations:")
        print(user_recs)
    except ValueError as error:
        print(f"Could not build user recommendations: {error}")
        user_recs = None

    print("\nSaving outputs...")
    save_outputs(top10_popular, similar_sample, user_recs)

    print("\nDone.")
    print(f"Saved files to: {DATA_DIR / 'model_a_outputs'}")


if __name__ == "__main__":
    main()
