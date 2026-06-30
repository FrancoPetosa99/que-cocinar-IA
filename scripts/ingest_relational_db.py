"""
Ingest recipes: CSV -> SQLite (full rows joined with Spanish enriched metadata).
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import SQLITE_PATH
from backend.recipe_db import ENRICHED_COLUMNS, SCHEMA, insert_recipe

RECIPES_CSV = ROOT / "data" / "recipes_spanish.csv"
RECIPES_FALLBACK_CSV = ROOT / "data" / "recipes.csv"
ENRICHED_CSV = ROOT / "data" / "enriched_recipes_spanish.csv"


def resolve_recipes_csv() -> Path:
    if RECIPES_CSV.exists():
        return RECIPES_CSV
    return RECIPES_FALLBACK_CSV


def load_joined_dataframe() -> pd.DataFrame:
    recipes_path = resolve_recipes_csv()
    if not recipes_path.exists():
        raise FileNotFoundError(f"Missing {recipes_path}")
    if not ENRICHED_CSV.exists():
        raise FileNotFoundError(f"Missing {ENRICHED_CSV}")

    recipes = pd.read_csv(recipes_path)
    enriched = pd.read_csv(ENRICHED_CSV)

    critical = ["recipe_name", "ingredients", "directions"]
    recipes = recipes.dropna(subset=critical).copy()
    recipes = recipes[recipes["recipe_name"].str.strip() != ""]
    recipes = recipes[recipes["ingredients"].str.strip() != ""]
    recipes = recipes[recipes["directions"].str.strip() != ""]

    if "Unnamed: 0" in recipes.columns:
        recipes["id"] = recipes["Unnamed: 0"].astype(int)
    else:
        recipes["id"] = recipes.index.astype(int)

    enriched = enriched.rename(columns={"recipe_id": "id"})
    merged = recipes.merge(enriched, on="id", how="left", suffixes=("", "_es"))

    if "recipe_name_es" in merged.columns:
        merged["recipe_name"] = merged["recipe_name"].combine_first(
            merged["recipe_name_es"]
        )
        merged = merged.drop(columns=["recipe_name_es"])

    return merged


def load_csv_to_sqlite(db_path: Path) -> int:
    df = load_joined_dataframe()

    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)

        for _, row in df.iterrows():
            insert_recipe(conn, row)

        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]

    return int(count)


def ingest_relational() -> int:
    recipes_path = resolve_recipes_csv()
    print(f"  Using recipes file: {recipes_path.name}")
    count = load_csv_to_sqlite(Path(SQLITE_PATH))
    print(f"  Inserted {count:,} recipes")
    enriched_count = load_joined_dataframe()[ENRICHED_COLUMNS[0]].notna().sum()
    print(f"  With Spanish enriched metadata: {enriched_count:,}")
    return count


def main() -> None:
    ingest_relational()
    print("\nDone. Restart the app: python frontend/app.py")


if __name__ == "__main__":
    main()
