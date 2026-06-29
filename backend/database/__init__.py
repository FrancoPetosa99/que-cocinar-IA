"""Read-only database facade for agents and frontend."""

from backend.database.read_only_interfaces import (
    RETRIEVAL_MAX_DISTANCE,
    Recipe,
    best_match_distance,
    get_connection,
    get_csv_row_preview,
    get_recipe_by_id,
    get_recipes_by_ids,
    get_vectorstore,
    reset_vectorstore,
    search_recipe_ids,
    search_recipe_ids_with_scores,
    search_recipes,
)

__all__ = [
    "RETRIEVAL_MAX_DISTANCE",
    "Recipe",
    "best_match_distance",
    "get_connection",
    "get_csv_row_preview",
    "get_recipe_by_id",
    "get_recipes_by_ids",
    "get_vectorstore",
    "reset_vectorstore",
    "search_recipe_ids",
    "search_recipe_ids_with_scores",
    "search_recipes",
]
