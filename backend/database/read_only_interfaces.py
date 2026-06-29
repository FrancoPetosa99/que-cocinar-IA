"""Read-only database interfaces for agents and pipeline (no ingestion logic)."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma

from backend.config import CHROMA_DIR, COLLECTION_NAME, SQLITE_PATH, get_embeddings

_vectorstore: Chroma | None = None

RETRIEVAL_MAX_DISTANCE = float(
    os.getenv(
        "RETRIEVAL_MAX_DISTANCE",
        os.getenv("RETRIEVAL_SCORE_THRESHOLD", "1.35"),
    )
)


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


def reset_vectorstore() -> None:
    global _vectorstore
    _vectorstore = None


def _validate_chroma_exists() -> None:
    chroma_path = Path(CHROMA_DIR)
    if not chroma_path.exists() or not any(chroma_path.iterdir()):
        raise FileNotFoundError(
            f"No se encontró la base vectorial en '{CHROMA_DIR}'. "
            "Ejecutá: python data_preprocessing/ingest.py"
        )


def _validate_collection(store: Chroma) -> None:
    try:
        count = store._collection.count()  # noqa: SLF001
        if count == 0:
            raise ValueError("La colección ChromaDB está vacía.")
        peek = store._collection.peek(1)  # noqa: SLF001
        metas = peek.get("metadatas") or []
        if not metas or "csv_row_id" not in (metas[0] or {}):
            raise ValueError(
                "El índice vectorial no tiene csv_row_id. "
                "Ejecutá: python data_preprocessing/ingest.py"
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "No se pudo validar chroma_db. Ejecutá: python data_preprocessing/ingest.py"
        ) from exc


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    _validate_chroma_exists()
    try:
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
        _validate_collection(_vectorstore)
    except Exception as exc:
        _vectorstore = None
        if isinstance(exc, ValueError):
            raise
        raise ConnectionError(
            f"No se pudo conectar a ChromaDB en '{CHROMA_DIR}': {exc}"
        ) from exc
    return _vectorstore


def _build_where_filter(
    *,
    max_total_time: int | None = None,
    min_protein: float | None = None,
    max_calories: float | None = None,
) -> dict[str, Any] | None:
    conditions: list[dict[str, Any]] = []
    if max_total_time is not None:
        conditions.append({"total_time_min": {"$lte": int(max_total_time)}})
    if min_protein is not None:
        conditions.append({"protein_g": {"$gte": float(min_protein)}})
    if max_calories is not None:
        conditions.append({"calories": {"$lte": float(max_calories)}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def search_recipe_ids_with_scores(
    query: str,
    *,
    max_total_time: int | None = None,
    min_protein: float | None = None,
    max_calories: float | None = None,
    k: int = 4,
    score_threshold: float | None = None,
) -> list[tuple[int, float]]:
    """Semantic search in Chroma. Returns (csv_row_id, distance) pairs."""
    threshold = score_threshold if score_threshold is not None else RETRIEVAL_MAX_DISTANCE
    where = _build_where_filter(
        max_total_time=max_total_time,
        min_protein=min_protein,
        max_calories=max_calories,
    )

    kwargs: dict[str, Any] = {"k": k}
    if where:
        kwargs["filter"] = where

    results = get_vectorstore().similarity_search_with_score(query, **kwargs)
    ids: list[tuple[int, float]] = []
    seen: set[int] = set()

    for doc, score in results:
        if score > threshold:
            continue
        row_id = doc.metadata.get("csv_row_id")
        if row_id is None:
            continue
        rid = int(row_id)
        if rid not in seen:
            seen.add(rid)
            ids.append((rid, score))
    return ids


def search_recipe_ids(
    query: str,
    *,
    max_total_time: int | None = None,
    min_protein: float | None = None,
    max_calories: float | None = None,
    k: int = 4,
    score_threshold: float | None = None,
) -> list[int]:
    """Return ordered recipe IDs from vector search (no full recipe data)."""
    return [
        rid
        for rid, _ in search_recipe_ids_with_scores(
            query,
            max_total_time=max_total_time,
            min_protein=min_protein,
            max_calories=max_calories,
            k=k,
            score_threshold=score_threshold,
        )
    ]


def best_match_distance(query: str) -> float | None:
    """Top-1 distance regardless of threshold (for debug hints)."""
    results = get_vectorstore().similarity_search_with_score(query, k=1)
    return results[0][1] if results else None


def search_recipes(
    query: str,
    *,
    max_total_time: int | None = None,
    min_protein: float | None = None,
    max_calories: float | None = None,
    k: int = 4,
) -> str:
    """Agent tool: vector search by ID, then load full rows from SQLite."""
    ids = search_recipe_ids(
        query,
        max_total_time=max_total_time,
        min_protein=min_protein,
        max_calories=max_calories,
        k=k,
    )
    if not ids:
        return "No matching recipes found in the database."

    recipes = get_recipes_by_ids(ids)
    blocks = []
    for recipe in recipes:
        blocks.append(
            f"[csv_row_id={recipe.id}] **{recipe.recipe_name}**\n"
            f"Time: {recipe.total_time_min or '?'} min | "
            f"Servings: {recipe.servings or '?'} | Rating: {recipe.rating or 'N/A'}\n\n"
            f"{recipe.embedding_text()}"
        )
    return "\n\n---\n\n".join(blocks)


def get_csv_row_preview(csv_row_id: int) -> str:
    """Return key fields for audit/testing."""
    recipe = get_recipe_by_id(csv_row_id)
    if recipe is None:
        return f"No row with csv_row_id={csv_row_id}"
    return (
        f"csv_row_id={recipe.id}\n"
        f"recipe_name={recipe.recipe_name}\n"
        f"ingredients={recipe.ingredients[:200]}...\n"
        f"directions={recipe.directions[:200]}..."
    )
