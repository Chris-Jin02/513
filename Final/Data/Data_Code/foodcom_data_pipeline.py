from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


RAW_FILE_URLS = {
    "RAW_recipes.csv": "https://huggingface.co/datasets/nutrientartcd/recipe-dataset/resolve/main/RAW_recipes.csv?download=true",
    "RAW_interactions.csv": "https://huggingface.co/datasets/nutrientartcd/recipe-dataset/resolve/main/RAW_interactions.csv?download=true",
}

NUTRITION_COLUMNS = [
    "calories",
    "total_fat_pdv",
    "sugar_pdv",
    "sodium_pdv",
    "protein_pdv",
    "saturated_fat_pdv",
    "carbohydrates_pdv",
]

TOKEN_CLEAN_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class PipelineConfig:
    workspace_root: Path
    min_user_ratings: int = 5
    min_recipe_ratings: int = 5
    force_download: bool = False

    @property
    def data_root(self) -> Path:
        return self.workspace_root / "Final" / "Data"

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "Raw_Data"

    @property
    def pure_dir(self) -> Path:
        return self.data_root / "Pure_Data"


def infer_workspace_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "Final").exists():
            return candidate
    raise FileNotFoundError("Could not locate the workspace root that contains a Final directory.")


def build_config(
    start: Path | None = None,
    *,
    min_user_ratings: int = 5,
    min_recipe_ratings: int = 5,
    force_download: bool = False,
) -> PipelineConfig:
    return PipelineConfig(
        workspace_root=infer_workspace_root(start),
        min_user_ratings=min_user_ratings,
        min_recipe_ratings=min_recipe_ratings,
        force_download=force_download,
    )


def ensure_directories(config: PipelineConfig) -> None:
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    config.pure_dir.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path, *, force: bool = False, chunk_size: int = 1024 * 1024) -> None:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"Skipping download, already exists: {destination.name}")
        return

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"Downloading {destination.name} ...")

    with urlopen(request) as response, destination.open("wb") as output_file:
        total_size = int(response.headers.get("Content-Length", "0"))
        downloaded = 0

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            output_file.write(chunk)
            downloaded += len(chunk)

            if total_size:
                progress = downloaded / total_size * 100
                if downloaded == total_size or downloaded % (50 * chunk_size) == 0:
                    print(f"  {destination.name}: {progress:5.1f}%")

    print(f"Finished: {destination}")


def download_raw_data(config: PipelineConfig) -> dict[str, Path]:
    ensure_directories(config)
    downloaded_paths: dict[str, Path] = {}

    for filename, url in RAW_FILE_URLS.items():
        destination = config.raw_dir / filename
        download_file(url, destination, force=config.force_download)
        downloaded_paths[filename] = destination

    return downloaded_paths


def parse_list_like(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []

    if isinstance(value, list):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            text = text.strip("[]")
            parsed = [item.strip().strip("'\"") for item in text.split(",") if item.strip()]

    if isinstance(parsed, tuple):
        parsed = list(parsed)
    elif not isinstance(parsed, list):
        parsed = [parsed]

    return [str(item).strip() for item in parsed if str(item).strip()]


def normalize_token(text: str) -> str:
    lowered = str(text).strip().lower()
    cleaned = TOKEN_CLEAN_PATTERN.sub("_", lowered).strip("_")
    return cleaned


def build_token_text(values: list[str]) -> str:
    tokens = [normalize_token(value) for value in values]
    return " ".join(token for token in tokens if token)


def build_free_text(values: list[str]) -> str:
    return " ".join(str(value).strip() for value in values if str(value).strip())


def expand_nutrition_column(series: pd.Series) -> pd.DataFrame:
    nutrition_values = series.apply(parse_list_like)
    expanded_rows = []

    for items in nutrition_values:
        row = []
        for index in range(len(NUTRITION_COLUMNS)):
            value = items[index] if index < len(items) else np.nan
            row.append(pd.to_numeric(value, errors="coerce"))
        expanded_rows.append(row)

    return pd.DataFrame(expanded_rows, columns=NUTRITION_COLUMNS, index=series.index)


def clean_recipes(recipes: pd.DataFrame) -> pd.DataFrame:
    cleaned = recipes.copy()
    cleaned = cleaned.drop_duplicates(subset=["id"]).reset_index(drop=True)

    cleaned["submitted"] = pd.to_datetime(cleaned["submitted"], errors="coerce")
    cleaned["description"] = cleaned["description"].fillna("").astype(str).str.strip()
    cleaned["name"] = cleaned["name"].fillna("").astype(str).str.strip()

    list_columns = {
        "tags": "tags_list",
        "ingredients": "ingredients_list",
        "steps": "steps_list",
    }
    for source_column, target_column in list_columns.items():
        cleaned[target_column] = cleaned[source_column].apply(parse_list_like)

    cleaned["tags_text"] = cleaned["tags_list"].apply(build_token_text)
    cleaned["ingredients_text"] = cleaned["ingredients_list"].apply(build_token_text)
    cleaned["steps_text"] = cleaned["steps_list"].apply(build_free_text)
    cleaned["combined_text"] = (
        cleaned["name"].fillna("")
        + " "
        + cleaned["description"].fillna("")
        + " "
        + cleaned["tags_text"].fillna("")
        + " "
        + cleaned["ingredients_text"].fillna("")
    ).str.strip()

    cleaned["parsed_tag_count"] = cleaned["tags_list"].str.len().astype("Int64")
    cleaned["parsed_ingredient_count"] = cleaned["ingredients_list"].str.len().astype("Int64")
    cleaned["parsed_step_count"] = cleaned["steps_list"].str.len().astype("Int64")

    nutrition_frame = expand_nutrition_column(cleaned["nutrition"])
    cleaned = pd.concat([cleaned, nutrition_frame], axis=1)

    cleaned["minutes"] = pd.to_numeric(cleaned["minutes"], errors="coerce")
    cleaned["n_steps"] = pd.to_numeric(cleaned["n_steps"], errors="coerce")
    cleaned["n_ingredients"] = pd.to_numeric(cleaned["n_ingredients"], errors="coerce")

    cleaned = cleaned[cleaned["id"].notna()].copy()
    cleaned = cleaned[cleaned["minutes"].fillna(0) > 0].copy()

    cleaned["submitted_year"] = cleaned["submitted"].dt.year.astype("Int64")
    cleaned["quick_recipe"] = cleaned["minutes"].le(30)

    list_export_columns = ["tags_list", "ingredients_list", "steps_list"]
    for column in list_export_columns:
        cleaned[column] = cleaned[column].apply(json.dumps)

    cleaned = cleaned.drop(columns=["tags", "nutrition", "ingredients", "steps"], errors="ignore")
    return cleaned.reset_index(drop=True)


def clean_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    cleaned = interactions.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["review"] = cleaned["review"].fillna("").astype(str).str.strip()
    cleaned["rating"] = pd.to_numeric(cleaned["rating"], errors="coerce")
    cleaned["user_id"] = pd.to_numeric(cleaned["user_id"], errors="coerce")
    cleaned["recipe_id"] = pd.to_numeric(cleaned["recipe_id"], errors="coerce")

    cleaned = cleaned.dropna(subset=["user_id", "recipe_id", "rating"]).copy()
    cleaned["user_id"] = cleaned["user_id"].astype("int64")
    cleaned["recipe_id"] = cleaned["recipe_id"].astype("int64")
    cleaned["rating"] = cleaned["rating"].astype("int64")

    cleaned = cleaned[cleaned["rating"].between(1, 5)].copy()
    cleaned = cleaned.sort_values(["user_id", "recipe_id", "date"], kind="mergesort")
    cleaned = cleaned.drop_duplicates(subset=["user_id", "recipe_id"], keep="last")
    cleaned = cleaned.reset_index(drop=True)

    return cleaned


def restrict_interactions_to_known_recipes(
    interactions: pd.DataFrame,
    recipes: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    valid_recipe_ids = set(recipes["id"].dropna().astype("int64"))
    filtered = interactions[interactions["recipe_id"].isin(valid_recipe_ids)].copy()
    removed_rows = int(len(interactions) - len(filtered))
    return filtered.reset_index(drop=True), removed_rows


def build_temporal_holdout_split(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = interactions.sort_values(["user_id", "date", "recipe_id"], kind="mergesort").reset_index(drop=True)
    user_counts = ordered.groupby("user_id")["recipe_id"].transform("size")
    eval_ready_mask = user_counts >= 2

    test = ordered[eval_ready_mask].groupby("user_id", group_keys=False).tail(1).copy()
    train = ordered.drop(index=test.index).reset_index(drop=True)

    return train, test.reset_index(drop=True)


def filter_interactions_by_support(
    interactions: pd.DataFrame,
    *,
    min_user_ratings: int,
    min_recipe_ratings: int,
) -> pd.DataFrame:
    filtered = interactions.copy()

    while True:
        starting_rows = len(filtered)
        user_counts = filtered["user_id"].value_counts()
        recipe_counts = filtered["recipe_id"].value_counts()

        valid_users = user_counts[user_counts >= min_user_ratings].index
        valid_recipes = recipe_counts[recipe_counts >= min_recipe_ratings].index

        filtered = filtered[
            filtered["user_id"].isin(valid_users) & filtered["recipe_id"].isin(valid_recipes)
        ].copy()

        if len(filtered) == starting_rows:
            break

    return filtered.reset_index(drop=True)


def filter_test_to_training_support(test_interactions: pd.DataFrame, train_interactions: pd.DataFrame) -> pd.DataFrame:
    valid_users = train_interactions["user_id"].unique()
    valid_recipes = train_interactions["recipe_id"].unique()

    filtered_test = test_interactions[
        test_interactions["user_id"].isin(valid_users) & test_interactions["recipe_id"].isin(valid_recipes)
    ].copy()

    return filtered_test.reset_index(drop=True)


def build_recipe_statistics(interactions: pd.DataFrame, *, prefix: str = "") -> pd.DataFrame:
    base_columns = ["recipe_id", "rating_count", "rating_mean", "rating_std", "last_interaction_date"]

    if interactions.empty:
        stats = pd.DataFrame(columns=base_columns)
    else:
        stats = (
            interactions.groupby("recipe_id")
            .agg(
                rating_count=("rating", "count"),
                rating_mean=("rating", "mean"),
                rating_std=("rating", "std"),
                last_interaction_date=("date", "max"),
            )
            .reset_index()
        )
        stats["rating_mean"] = stats["rating_mean"].round(4)
        stats["rating_std"] = stats["rating_std"].round(4)

    if not prefix:
        return stats

    rename_map = {
        "rating_count": f"{prefix}_rating_count",
        "rating_mean": f"{prefix}_rating_mean",
        "rating_std": f"{prefix}_rating_std",
        "last_interaction_date": f"{prefix}_last_interaction_date",
    }
    return stats.rename(columns=rename_map)


def build_user_statistics_frame(interactions: pd.DataFrame, *, prefix: str = "") -> pd.DataFrame:
    base_columns = ["user_id", "user_rating_count", "user_rating_mean", "last_interaction_date"]

    if interactions.empty:
        stats = pd.DataFrame(columns=base_columns)
    else:
        stats = (
            interactions.groupby("user_id")
            .agg(
                user_rating_count=("rating", "count"),
                user_rating_mean=("rating", "mean"),
                last_interaction_date=("date", "max"),
            )
            .reset_index()
        )
        stats["user_rating_mean"] = stats["user_rating_mean"].round(4)

    if not prefix:
        return stats

    rename_map = {
        "user_rating_count": f"{prefix}_user_rating_count",
        "user_rating_mean": f"{prefix}_user_rating_mean",
        "last_interaction_date": f"{prefix}_last_interaction_date",
    }
    return stats.rename(columns=rename_map)


def build_recipe_history_bucket(counts: pd.Series, *, min_recipe_ratings: int) -> pd.Series:
    counts_int = counts.fillna(0).astype("int64")
    labels = np.where(
        counts_int == 0,
        "no_history",
        np.where(counts_int < min_recipe_ratings, "low_history", "enough_history"),
    )
    return pd.Series(labels, index=counts.index, dtype="object")


def build_user_history_bucket(counts: pd.Series, *, min_user_ratings: int) -> pd.Series:
    counts_int = counts.fillna(0).astype("int64")
    labels = np.where(
        counts_int == 0,
        "no_history",
        np.where(
            counts_int == 1,
            "single_interaction",
            np.where(counts_int < min_user_ratings, "low_history", "enough_history"),
        ),
    )
    return pd.Series(labels, index=counts.index, dtype="object")


def build_recipe_model_table(
    recipes: pd.DataFrame,
    interactions_clean: pd.DataFrame,
    interactions_filtered: pd.DataFrame,
    *,
    min_recipe_ratings: int,
) -> pd.DataFrame:
    clean_stats = build_recipe_statistics(interactions_clean, prefix="clean")
    filtered_stats = build_recipe_statistics(interactions_filtered, prefix="filtered")

    model_columns = [
        "id",
        "name",
        "minutes",
        "contributor_id",
        "submitted",
        "submitted_year",
        "description",
        "n_steps",
        "n_ingredients",
        "parsed_tag_count",
        "parsed_ingredient_count",
        "parsed_step_count",
        "quick_recipe",
        "tags_list",
        "ingredients_list",
        "tags_text",
        "ingredients_text",
        "combined_text",
        *NUTRITION_COLUMNS,
    ]
    model_table = recipes[model_columns].merge(clean_stats, left_on="id", right_on="recipe_id", how="left")
    model_table = model_table.merge(filtered_stats, left_on="id", right_on="recipe_id", how="left")

    recipe_id_columns = [column for column in model_table.columns if column.startswith("recipe_id")]
    model_table = model_table.drop(columns=recipe_id_columns, errors="ignore")

    for prefix in ["clean", "filtered"]:
        count_column = f"{prefix}_rating_count"
        mean_column = f"{prefix}_rating_mean"
        std_column = f"{prefix}_rating_std"

        model_table[count_column] = model_table[count_column].fillna(0).astype("int64")
        model_table[mean_column] = model_table[mean_column].fillna(0.0)
        model_table[std_column] = model_table[std_column].fillna(0.0)

    model_table["rating_count"] = model_table["clean_rating_count"]
    model_table["rating_mean"] = model_table["clean_rating_mean"]
    model_table["rating_std"] = model_table["clean_rating_std"]
    model_table["last_interaction_date"] = model_table["clean_last_interaction_date"]
    model_table["has_clean_history"] = model_table["clean_rating_count"] > 0
    model_table["has_filtered_history"] = model_table["filtered_rating_count"] > 0
    model_table["history_bucket"] = build_recipe_history_bucket(
        model_table["clean_rating_count"],
        min_recipe_ratings=min_recipe_ratings,
    )
    model_table["has_enough_history"] = model_table["clean_rating_count"] >= min_recipe_ratings
    model_table["eligible_for_cf"] = model_table["filtered_rating_count"] > 0

    return model_table


def build_user_statistics(
    interactions_clean: pd.DataFrame,
    interactions_filtered: pd.DataFrame,
    *,
    min_user_ratings: int,
) -> pd.DataFrame:
    clean_stats = build_user_statistics_frame(interactions_clean, prefix="clean")
    filtered_stats = build_user_statistics_frame(interactions_filtered, prefix="filtered")

    user_stats = clean_stats.merge(filtered_stats, on="user_id", how="left")

    for prefix in ["clean", "filtered"]:
        count_column = f"{prefix}_user_rating_count"
        mean_column = f"{prefix}_user_rating_mean"

        user_stats[count_column] = user_stats[count_column].fillna(0).astype("int64")
        user_stats[mean_column] = user_stats[mean_column].fillna(0.0)

    user_stats["user_rating_count"] = user_stats["clean_user_rating_count"]
    user_stats["user_rating_mean"] = user_stats["clean_user_rating_mean"]
    user_stats["last_interaction_date"] = user_stats["clean_last_interaction_date"]
    user_stats["history_bucket"] = build_user_history_bucket(
        user_stats["clean_user_rating_count"],
        min_user_ratings=min_user_ratings,
    )
    user_stats["has_enough_history"] = user_stats["clean_user_rating_count"] >= min_user_ratings
    user_stats["eligible_for_cf"] = user_stats["filtered_user_rating_count"] > 0

    return user_stats


def summarize_pipeline(
    recipes_raw: pd.DataFrame,
    interactions_raw: pd.DataFrame,
    recipes_clean: pd.DataFrame,
    interactions_clean_before_recipe_alignment: pd.DataFrame,
    interactions_clean: pd.DataFrame,
    interactions_filtered: pd.DataFrame,
    recipe_model_table: pd.DataFrame,
    user_statistics: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, Any]:
    return {
        "raw_recipe_rows": int(len(recipes_raw)),
        "raw_interaction_rows": int(len(interactions_raw)),
        "clean_recipe_rows": int(len(recipes_clean)),
        "clean_interaction_rows_before_recipe_alignment": int(len(interactions_clean_before_recipe_alignment)),
        "clean_interaction_rows": int(len(interactions_clean)),
        "dropped_interactions_missing_clean_recipe": int(
            len(interactions_clean_before_recipe_alignment) - len(interactions_clean)
        ),
        "clean_user_count": int(interactions_clean["user_id"].nunique()),
        "clean_interacted_recipe_count": int(interactions_clean["recipe_id"].nunique()),
        "filtered_interaction_rows": int(len(interactions_filtered)),
        "filtered_recipe_count": int(interactions_filtered["recipe_id"].nunique()),
        "filtered_user_count": int(interactions_filtered["user_id"].nunique()),
        "recipes_with_clean_history": int(recipe_model_table["has_clean_history"].sum()),
        "recipes_eligible_for_cf": int(recipe_model_table["eligible_for_cf"].sum()),
        "users_eligible_for_cf": int(user_statistics["eligible_for_cf"].sum()),
        "min_user_ratings": config.min_user_ratings,
        "min_recipe_ratings": config.min_recipe_ratings,
        "output_files": [
            "recipes_clean.csv",
            "interactions_clean.csv",
            "interactions_filtered.csv",
            "interactions_train.csv",
            "interactions_test.csv",
            "interactions_train_filtered.csv",
            "interactions_test_filtered.csv",
            "recipe_model_table.csv",
            "user_statistics.csv",
            "preprocessing_summary.json",
            "temporal_split_summary.json",
        ],
    }


def summarize_temporal_split(
    interactions_clean: pd.DataFrame,
    interactions_train: pd.DataFrame,
    interactions_test: pd.DataFrame,
    interactions_train_filtered: pd.DataFrame,
    interactions_test_filtered: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, Any]:
    clean_user_counts = interactions_clean.groupby("user_id").size()
    train_user_counts = interactions_train.groupby("user_id").size()
    train_recipe_counts = interactions_train.groupby("recipe_id").size()

    return {
        "split_strategy": "per_user_temporal_holdout_last_interaction",
        "eligible_eval_user_count": int((clean_user_counts >= 2).sum()),
        "single_interaction_user_count": int((clean_user_counts == 1).sum()),
        "train_interaction_rows": int(len(interactions_train)),
        "test_interaction_rows": int(len(interactions_test)),
        "train_filtered_interaction_rows": int(len(interactions_train_filtered)),
        "train_filtered_user_count": int(interactions_train_filtered["user_id"].nunique()),
        "train_filtered_recipe_count": int(interactions_train_filtered["recipe_id"].nunique()),
        "test_filtered_interaction_rows": int(len(interactions_test_filtered)),
        "test_filtered_user_count": int(interactions_test_filtered["user_id"].nunique()),
        "test_filtered_recipe_count": int(interactions_test_filtered["recipe_id"].nunique()),
        "users_below_min_support_in_train": int((train_user_counts < config.min_user_ratings).sum()),
        "recipes_below_min_support_in_train": int((train_recipe_counts < config.min_recipe_ratings).sum()),
        "test_rows_dropped_for_cf_evaluation": int(len(interactions_test) - len(interactions_test_filtered)),
    }


def save_outputs(
    config: PipelineConfig,
    recipes_clean: pd.DataFrame,
    interactions_clean: pd.DataFrame,
    interactions_filtered: pd.DataFrame,
    interactions_train: pd.DataFrame,
    interactions_test: pd.DataFrame,
    interactions_train_filtered: pd.DataFrame,
    interactions_test_filtered: pd.DataFrame,
    recipe_model_table: pd.DataFrame,
    user_statistics: pd.DataFrame,
    summary: dict[str, Any],
    temporal_split_summary: dict[str, Any],
) -> None:
    ensure_directories(config)

    output_frames = {
        "recipes_clean.csv": recipes_clean,
        "interactions_clean.csv": interactions_clean,
        "interactions_filtered.csv": interactions_filtered,
        "interactions_train.csv": interactions_train,
        "interactions_test.csv": interactions_test,
        "interactions_train_filtered.csv": interactions_train_filtered,
        "interactions_test_filtered.csv": interactions_test_filtered,
        "recipe_model_table.csv": recipe_model_table,
        "user_statistics.csv": user_statistics,
    }

    for filename, frame in output_frames.items():
        frame.to_csv(config.pure_dir / filename, index=False)

    with (config.pure_dir / "preprocessing_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, default=str)

    with (config.pure_dir / "temporal_split_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(temporal_split_summary, output_file, indent=2, default=str)


def run_pipeline(config: PipelineConfig | None = None) -> dict[str, Any]:
    active_config = config or build_config()
    ensure_directories(active_config)
    raw_files = download_raw_data(active_config)

    print("Loading raw CSV files ...")
    recipes_raw = pd.read_csv(raw_files["RAW_recipes.csv"], low_memory=False)
    interactions_raw = pd.read_csv(raw_files["RAW_interactions.csv"], low_memory=False)

    print("Cleaning recipes table ...")
    recipes_clean = clean_recipes(recipes_raw)

    print("Cleaning interactions table ...")
    interactions_clean_all = clean_interactions(interactions_raw)

    print("Restricting interactions to cleaned recipe ids ...")
    interactions_clean, removed_interactions_missing_recipe = restrict_interactions_to_known_recipes(
        interactions_clean_all,
        recipes_clean,
    )
    print(f"Removed {removed_interactions_missing_recipe} interactions whose recipe ids are not in recipes_clean.")

    print("Building temporal train/test split ...")
    interactions_train, interactions_test = build_temporal_holdout_split(interactions_clean)

    print("Applying support thresholds to the full interaction table ...")
    interactions_filtered = filter_interactions_by_support(
        interactions_clean,
        min_user_ratings=active_config.min_user_ratings,
        min_recipe_ratings=active_config.min_recipe_ratings,
    )

    print("Applying support thresholds to the training split ...")
    interactions_train_filtered = filter_interactions_by_support(
        interactions_train,
        min_user_ratings=active_config.min_user_ratings,
        min_recipe_ratings=active_config.min_recipe_ratings,
    )
    interactions_test_filtered = filter_test_to_training_support(interactions_test, interactions_train_filtered)

    print("Building recipe modeling table ...")
    recipe_model_table = build_recipe_model_table(
        recipes_clean,
        interactions_clean,
        interactions_filtered,
        min_recipe_ratings=active_config.min_recipe_ratings,
    )

    print("Building user statistics table ...")
    user_stats = build_user_statistics(
        interactions_clean,
        interactions_filtered,
        min_user_ratings=active_config.min_user_ratings,
    )

    summary = summarize_pipeline(
        recipes_raw=recipes_raw,
        interactions_raw=interactions_raw,
        recipes_clean=recipes_clean,
        interactions_clean_before_recipe_alignment=interactions_clean_all,
        interactions_clean=interactions_clean,
        interactions_filtered=interactions_filtered,
        recipe_model_table=recipe_model_table,
        user_statistics=user_stats,
        config=active_config,
    )
    temporal_split_summary = summarize_temporal_split(
        interactions_clean=interactions_clean,
        interactions_train=interactions_train,
        interactions_test=interactions_test,
        interactions_train_filtered=interactions_train_filtered,
        interactions_test_filtered=interactions_test_filtered,
        config=active_config,
    )

    print("Saving outputs ...")
    save_outputs(
        active_config,
        recipes_clean=recipes_clean,
        interactions_clean=interactions_clean,
        interactions_filtered=interactions_filtered,
        interactions_train=interactions_train,
        interactions_test=interactions_test,
        interactions_train_filtered=interactions_train_filtered,
        interactions_test_filtered=interactions_test_filtered,
        recipe_model_table=recipe_model_table,
        user_statistics=user_stats,
        summary=summary,
        temporal_split_summary=temporal_split_summary,
    )

    print("Pipeline complete.")
    return {
        "config": active_config,
        "raw_files": raw_files,
        "recipes_clean": recipes_clean,
        "interactions_clean": interactions_clean,
        "interactions_filtered": interactions_filtered,
        "interactions_train": interactions_train,
        "interactions_test": interactions_test,
        "interactions_train_filtered": interactions_train_filtered,
        "interactions_test_filtered": interactions_test_filtered,
        "recipe_model_table": recipe_model_table,
        "user_statistics": user_stats,
        "summary": summary,
        "temporal_split_summary": temporal_split_summary,
    }


if __name__ == "__main__":
    results = run_pipeline()
    print(json.dumps(results["summary"], indent=2, default=str))
    print(json.dumps(results["temporal_split_summary"], indent=2, default=str))
