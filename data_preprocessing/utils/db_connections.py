"""Write-time SQLite paths and ingestion helpers (no backend imports)."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from data_preprocessing.utils.cleaning import (
    clean_recipes_dataframe,
    nullable_str,
    parse_nutrition,
    parse_servings,
    time_to_minutes,
)

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_CSV_PATH = PROJECT_ROOT / "data_preprocessing" / "raw_data" / "recipes.csv"
ENRICHED_CSV_PATH = PROJECT_ROOT / "data_preprocessing" / "raw_data" / "enriched_recipes.csv"

_chroma_env = os.getenv("CHROMA_DIR")
if _chroma_env:
    _chroma_path = Path(_chroma_env)
    CHROMA_DIR = str(
        _chroma_path if _chroma_path.is_absolute() else PROJECT_ROOT / _chroma_path
    )
else:
    CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "recipes")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

_sqlite_env = os.getenv("SQLITE_PATH")
if _sqlite_env:
    _sqlite_path = Path(_sqlite_env)
    SQLITE_PATH = str(
        _sqlite_path if _sqlite_path.is_absolute() else PROJECT_ROOT / _sqlite_path
    )
else:
    SQLITE_PATH = str(PROJECT_ROOT / "data" / "recipes.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS recipes (
    id              INTEGER PRIMARY KEY,
    recipe_name     TEXT NOT NULL,
    prep_time       TEXT,
    cook_time       TEXT,
    total_time      TEXT,
    prep_time_min   INTEGER,
    cook_time_min   INTEGER,
    total_time_min  INTEGER,
    servings        INTEGER,
    yield_text      TEXT,
    ingredients     TEXT NOT NULL,
    directions      TEXT NOT NULL,
    rating          REAL,
    url             TEXT,
    cuisine_path    TEXT,
    nutrition       TEXT,
    timing          TEXT,
    img_src         TEXT,
    calories        REAL,
    protein_g       REAL,
    carbs_g         REAL,
    fat_g           REAL
);
"""


def get_sqlite_write_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path or SQLITE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def load_csv_to_sqlite(
    csv_path: Path | str | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Load recipes.csv into SQLite. Returns number of rows inserted."""
    csv_path = Path(csv_path or RAW_CSV_PATH)
    db_path = Path(db_path or SQLITE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = clean_recipes_dataframe(pd.read_csv(csv_path))

    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        for _, row in df.iterrows():
            nutrition = parse_nutrition(row.get("nutrition"))
            conn.execute(
                """
                INSERT INTO recipes (
                    id, recipe_name, prep_time, cook_time, total_time,
                    prep_time_min, cook_time_min, total_time_min,
                    servings, yield_text, ingredients, directions,
                    rating, url, cuisine_path, nutrition, timing, img_src,
                    calories, protein_g, carbs_g, fat_g
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    int(row["id"]),
                    str(row["recipe_name"]).strip(),
                    nullable_str(row.get("prep_time")),
                    nullable_str(row.get("cook_time")),
                    nullable_str(row.get("total_time")),
                    time_to_minutes(row.get("prep_time")),
                    time_to_minutes(row.get("cook_time")),
                    time_to_minutes(row.get("total_time")),
                    parse_servings(row.get("servings")),
                    nullable_str(row.get("yield")),
                    str(row["ingredients"]).strip(),
                    str(row["directions"]).strip(),
                    float(row["rating"]) if pd.notna(row.get("rating")) else None,
                    nullable_str(row.get("url")),
                    nullable_str(row.get("cuisine_path")),
                    nullable_str(row.get("nutrition")),
                    nullable_str(row.get("timing")),
                    nullable_str(row.get("img_src")),
                    nutrition.get("calories"),
                    nutrition.get("protein_g"),
                    nutrition.get("carbs_g"),
                    nutrition.get("fat_g"),
                ),
            )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]

    return int(count)


def iter_recipe_rows(db_path: Path | str | None = None) -> list[sqlite3.Row]:
    """Return all recipe rows from SQLite for Chroma indexing."""
    path = Path(db_path or SQLITE_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró la base relacional en '{path}'. "
            "Ejecutá primero la fase relacional del ingest."
        )
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM recipes ORDER BY id").fetchall()
