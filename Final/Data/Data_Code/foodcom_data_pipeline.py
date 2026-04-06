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


def build_recipe_statistics(interactions: pd.DataFrame) -> pd.DataFrame:
    recipe_stats = (
        interactions.groupby("recipe_id")
        .agg(
            rating_count=("rating", "count"),
            rating_mean=("rating", "mean"),
            rating_std=("rating", "std"),
            last_interaction_date=("date", "max"),
        )
        .reset_index()
    )

    recipe_stats["rating_mean"] = recipe_stats["rating_mean"].round(4)
    recipe_stats["rating_std"] = recipe_stats["rating_std"].round(4)
    return recipe_stats


def build_user_statistics(interactions: pd.DataFrame) -> pd.DataFrame:
    user_stats = (
        interactions.groupby("user_id")
        .agg(
            user_rating_count=("rating", "count"),
            user_rating_mean=("rating", "mean"),
        )
        .reset_index()
    )
    user_stats["user_rating_mean"] = user_stats["user_rating_mean"].round(4)
    return user_stats


def build_recipe_model_table(recipes: pd.DataFrame, interactions_filtered: pd.DataFrame) -> pd.DataFrame:
    recipe_stats = build_recipe_statistics(interactions_filtered)
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
    model_table = recipes[model_columns].merge(recipe_stats, left_on="id", right_on="recipe_id", how="left")

    model_table["rating_count"] = model_table["rating_count"].fillna(0).astype("int64")
    model_table["rating_mean"] = model_table["rating_mean"].fillna(0.0)
    model_table["rating_std"] = model_table["rating_std"].fillna(0.0)
    model_table["has_enough_history"] = model_table["rating_count"] > 0

    return model_table.drop(columns=["recipe_id"], errors="ignore")


def save_outputs(
    config: PipelineConfig,
    recipes_clean: pd.DataFrame,
    interactions_clean: pd.DataFrame,
    interactions_filtered: pd.DataFrame,
    recipe_model_table: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    ensure_directories(config)

    recipes_clean.to_csv(config.pure_dir / "recipes_clean.csv", index=False)
    interactions_clean.to_csv(config.pure_dir / "interactions_clean.csv", index=False)
    interactions_filtered.to_csv(config.pure_dir / "interactions_filtered.csv", index=False)
    recipe_model_table.to_csv(config.pure_dir / "recipe_model_table.csv", index=False)

    with (config.pure_dir / "preprocessing_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, default=str)


def summarize_pipeline(
    recipes_raw: pd.DataFrame,
    interactions_raw: pd.DataFrame,
    recipes_clean: pd.DataFrame,
    interactions_clean: pd.DataFrame,
    interactions_filtered: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, Any]:
    return {
        "raw_recipe_rows": int(len(recipes_raw)),
        "raw_interaction_rows": int(len(interactions_raw)),
        "clean_recipe_rows": int(len(recipes_clean)),
        "clean_interaction_rows": int(len(interactions_clean)),
        "filtered_interaction_rows": int(len(interactions_filtered)),
        "filtered_recipe_count": int(interactions_filtered["recipe_id"].nunique()),
        "filtered_user_count": int(interactions_filtered["user_id"].nunique()),
        "min_user_ratings": config.min_user_ratings,
        "min_recipe_ratings": config.min_recipe_ratings,
        "output_files": [
            "recipes_clean.csv",
            "interactions_clean.csv",
            "interactions_filtered.csv",
            "recipe_model_table.csv",
            "preprocessing_summary.json",
        ],
    }


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
    interactions_clean = clean_interactions(interactions_raw)

    print("Applying collaborative-filtering support thresholds ...")
    interactions_filtered = filter_interactions_by_support(
        interactions_clean,
        min_user_ratings=active_config.min_user_ratings,
        min_recipe_ratings=active_config.min_recipe_ratings,
    )

    print("Building recipe modeling table ...")
    recipe_model_table = build_recipe_model_table(recipes_clean, interactions_filtered)
    user_stats = build_user_statistics(interactions_filtered)
    user_stats.to_csv(active_config.pure_dir / "user_statistics.csv", index=False)

    summary = summarize_pipeline(
        recipes_raw=recipes_raw,
        interactions_raw=interactions_raw,
        recipes_clean=recipes_clean,
        interactions_clean=interactions_clean,
        interactions_filtered=interactions_filtered,
        config=active_config,
    )
    summary["output_files"].append("user_statistics.csv")

    print("Saving outputs ...")
    save_outputs(
        active_config,
        recipes_clean=recipes_clean,
        interactions_clean=interactions_clean,
        interactions_filtered=interactions_filtered,
        recipe_model_table=recipe_model_table,
        summary=summary,
    )

    print("Pipeline complete.")
    return {
        "config": active_config,
        "raw_files": raw_files,
        "recipes_clean": recipes_clean,
        "interactions_clean": interactions_clean,
        "interactions_filtered": interactions_filtered,
        "recipe_model_table": recipe_model_table,
        "user_statistics": user_stats,
        "summary": summary,
    }


if __name__ == "__main__":
    results = run_pipeline()
    print(json.dumps(results["summary"], indent=2, default=str))
