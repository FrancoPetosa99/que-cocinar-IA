"""SQLite relational store — full recipe rows (source of truth)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import PROJECT_ROOT, SQLITE_PATH
from backend.recipe_parsing import parse_nutrition, parse_servings, time_to_minutes

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

@dataclass
class Recipe:
    """Full recipe row from SQLite."""

    id: int
    recipe_name: str
    ingredients: str
    directions: str
    prep_time: str | None = None
    cook_time: str | None = None
    total_time: str | None = None
    prep_time_min: int | None = None
    cook_time_min: int | None = None
    total_time_min: int | None = None
    servings: int | None = None
    yield_text: str | None = None
    rating: float | None = None
    url: str | None = None
    cuisine_path: str | None = None
    nutrition: str | None = None
    timing: str | None = None
    img_src: str | None = None
    calories: float | None = None
    protein_g: float | None = None
    carbs_g: float | None = None
    fat_g: float | None = None

    def embedding_text(self) -> str:
        """Text indexed in Chroma (name + ingredients only)."""
        return f"{self.recipe_name}\n\nIngredients:\n{self.ingredients}"

    def full_text(self) -> str:
        """Full recipe text for scaling tools."""
        return (
            f"{self.recipe_name}\n\n"
            f"Ingredients:\n{self.ingredients}\n\n"
            f"Directions:\n{self.directions}"
        )

def _row_to_recipe(row: sqlite3.Row) -> Recipe:
    return Recipe(
        id=int(row["id"]),
        recipe_name=row["recipe_name"],
        ingredients=row["ingredients"],
        directions=row["directions"],
        prep_time=row["prep_time"],
        cook_time=row["cook_time"],
        total_time=row["total_time"],
        prep_time_min=row["prep_time_min"],
        cook_time_min=row["cook_time_min"],
        total_time_min=row["total_time_min"],
        servings=row["servings"],
        yield_text=row["yield_text"],
        rating=row["rating"],
        url=row["url"],
        cuisine_path=row["cuisine_path"],
        nutrition=row["nutrition"],
        timing=row["timing"],
        img_src=row["img_src"],
        calories=row["calories"],
        protein_g=row["protein_g"],
        carbs_g=row["carbs_g"],
        fat_g=row["fat_g"],
    )

def get_connection() -> sqlite3.Connection:
    db_path = Path(SQLITE_PATH)
    if not db_path.exists():
        raise FileNotFoundError(
            f"No se encontró la base relacional en '{db_path}'. "
            "Ejecutá: python data_preprocessing/ingest.py"
        )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def get_recipe_by_id(recipe_id: int) -> Recipe | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM recipes WHERE id = ?", (recipe_id,)
        ).fetchone()
    return _row_to_recipe(row) if row else None

def get_recipes_by_ids(recipe_ids: list[int]) -> list[Recipe]:
    """Fetch recipes preserving the order of recipe_ids."""
    if not recipe_ids:
        return []
    placeholders = ",".join("?" * len(recipe_ids))
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM recipes WHERE id IN ({placeholders})",
            recipe_ids,
        ).fetchall()
    by_id = {_row_to_recipe(r).id: _row_to_recipe(r) for r in rows}
    return [by_id[i] for i in recipe_ids if i in by_id]

def insert_recipe(conn: sqlite3.Connection, row: pd.Series) -> None:
    """Inserta una receta en la base de datos."""

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
            _nullable_str(row.get("prep_time")),
            _nullable_str(row.get("cook_time")),
            _nullable_str(row.get("total_time")),
            time_to_minutes(row.get("prep_time")),
            time_to_minutes(row.get("cook_time")),
            time_to_minutes(row.get("total_time")),
            parse_servings(row.get("servings")),
            _nullable_str(row.get("yield")),
            str(row["ingredients"]).strip(),
            str(row["directions"]).strip(),
            float(row["rating"]) if pd.notna(row.get("rating")) else None,
            _nullable_str(row.get("url")),
            _nullable_str(row.get("cuisine_path")),
            _nullable_str(row.get("nutrition")),
            _nullable_str(row.get("timing")),
            _nullable_str(row.get("img_src")),
            nutrition.get("calories"),
            nutrition.get("protein_g"),
            nutrition.get("carbs_g"),
            nutrition.get("fat_g"),
        ),
    )

def _nullable_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None