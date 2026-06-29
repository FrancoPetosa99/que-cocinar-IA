"""Shared ingestion utilities for preprocessing pipelines."""

from data_preprocessing.utils.cleaning import (
    clean_recipes_dataframe,
    nullable_str,
    parse_nutrition,
    parse_servings,
    time_to_minutes,
)
from data_preprocessing.utils.db_connections import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    ENRICHED_CSV_PATH,
    PROJECT_ROOT,
    RAW_CSV_PATH,
    SCHEMA,
    SQLITE_PATH,
    get_sqlite_write_connection,
    iter_recipe_rows,
    load_csv_to_sqlite,
)
from data_preprocessing.utils.vector_build import (
    build_chroma_index,
    build_chroma_index_from_sqlite,
    recipe_row_to_document,
    rows_to_documents,
)

__all__ = [
    "CHROMA_DIR",
    "COLLECTION_NAME",
    "EMBEDDING_MODEL",
    "ENRICHED_CSV_PATH",
    "PROJECT_ROOT",
    "RAW_CSV_PATH",
    "SCHEMA",
    "SQLITE_PATH",
    "build_chroma_index",
    "build_chroma_index_from_sqlite",
    "clean_recipes_dataframe",
    "get_sqlite_write_connection",
    "iter_recipe_rows",
    "load_csv_to_sqlite",
    "nullable_str",
    "parse_nutrition",
    "parse_servings",
    "recipe_row_to_document",
    "rows_to_documents",
    "time_to_minutes",
]
