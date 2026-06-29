"""
Ingest recipes: CSV -> SQLite (full rows) + Chroma (search index only).

"""

from __future__ import annotations

import sys
import pandas as pd
from pathlib import Path
import sqlite3

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import SQLITE_PATH
from backend.recipe_db import SCHEMA, insert_recipe
from backend.recipe_parsing import parse_nutrition, parse_servings, time_to_minutes

CSV_PATH = ROOT / "data" / "recipes.csv"

def ingest_relational() -> int:
    count = load_csv_to_sqlite(CSV_PATH, Path(SQLITE_PATH))
    print(f"  Inserted {count:,} recipes")
    return count

def load_csv_to_sqlite(csv_path: Path, db_path: Path) -> int:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)

    critical = ["recipe_name", "ingredients", "directions"]

    df = df.dropna(subset=critical).copy()
    df = df[df["recipe_name"].str.strip() != ""]
    df = df[df["ingredients"].str.strip() != ""]
    df = df[df["directions"].str.strip() != ""]

    if "Unnamed: 0" in df.columns:
        df["id"] = df["Unnamed: 0"].astype(int)
    else:
        df["id"] = df.index.astype(int)

    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:

        conn.executescript(SCHEMA)

        for _, row in df.iterrows():
            insert_recipe(conn, row)

        conn.commit()

        count = conn.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]

    return int(count)

def main() -> None:
    ingest_relational()
    print("\nDone. Restart the app: python frontend/app.py")

if __name__ == "__main__":
    main()
