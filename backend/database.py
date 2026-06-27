"""Facade: vector search (IDs) + relational fetch (full rows)."""

from __future__ import annotations

from backend.recipe_db import Recipe, get_recipe_by_id, get_recipes_by_ids
from backend.vector_store import (
    RETRIEVAL_MAX_DISTANCE,
    best_match_distance,
    get_vectorstore,
    recipe_to_vector_document,
    reset_vectorstore,
    search_recipe_ids,
    search_recipe_ids_with_scores,
)

__all__ = [
    "RETRIEVAL_MAX_DISTANCE",
    "Recipe",
    "best_match_distance",
    "get_csv_row_preview",
    "get_recipe_by_id",
    "get_recipes_by_ids",
    "get_vectorstore",
    "recipe_to_vector_document",
    "reset_vectorstore",
    "search_recipe_ids",
    "search_recipe_ids_with_scores",
    "search_recipes",
]


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
