#!/usr/bin/env python3
"""Rebuild chroma_db from data/recipes.csv (includes csv_row_id in metadata)."""

from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "recipes.csv"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "recipes"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 500


def time_to_minutes(value) -> int | None:
    if pd.isna(value) or value == "":
        return None
    text = str(value).lower().strip()
    hours = minutes = 0
    hr_match = re.search(r"(\d+)\s*h(?:r|our|ora)?", text)
    min_match = re.search(r"(\d+)\s*m(?:in|inute|inuto)?", text)
    if hr_match:
        hours = int(hr_match.group(1))
    if min_match:
        minutes = int(min_match.group(1))
    if not hr_match and not min_match:
        digits = re.findall(r"\d+", text)
        return int(digits[0]) if digits else None
    return hours * 60 + minutes


def parse_servings(value) -> int | None:
    if pd.isna(value) or value == "":
        return None
    digits = re.findall(r"\d+", str(value))
    return int(digits[0]) if digits else None


def parse_nutrition(value) -> dict:
    result = {"calories": None, "protein_g": None, "carbs_g": None, "fat_g": None}
    if pd.isna(value) or value == "":
        return result
    try:
        data = ast.literal_eval(str(value)) if isinstance(value, str) else value
    except (ValueError, SyntaxError):
        return result
    if not isinstance(data, dict):
        return result
    key_map = {
        "calories": "calories",
        "proteinContent": "protein_g",
        "carbohydrateContent": "carbs_g",
        "fatContent": "fat_g",
    }
    for src, dst in key_map.items():
        raw = data.get(src)
        if raw is None:
            continue
        nums = re.findall(r"[\d.]+", str(raw))
        if nums:
            result[dst] = float(nums[0])
    return result


def row_to_document(row) -> Document:
    name = str(row["recipe_name"]).strip()
    ingredients = str(row["ingredients"]).strip()
    directions = str(row["directions"]).strip()
    page_content = (
        f"{name}\n\n"
        f"Ingredients:\n{ingredients}\n\n"
        f"Directions:\n{directions}"
    )
    nutrition = parse_nutrition(row.get("nutrition"))
    metadata = {
        "csv_row_id": int(row["csv_row_id"]),
        "recipe_name": name,
        "prep_time_min": time_to_minutes(row.get("prep_time")),
        "cook_time_min": time_to_minutes(row.get("cook_time")),
        "total_time_min": time_to_minutes(row.get("total_time")),
        "servings": parse_servings(row.get("servings")),
        "rating": float(row["rating"]) if pd.notna(row.get("rating")) else None,
        "cuisine_path": str(row.get("cuisine_path", ""))
        if pd.notna(row.get("cuisine_path"))
        else "",
        **nutrition,
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}
    return Document(page_content=page_content, metadata=metadata)


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}")

    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    critical = ["recipe_name", "ingredients", "directions"]
    df = df.dropna(subset=critical).copy()
    df = df[df["recipe_name"].str.strip() != ""]
    df = df[df["ingredients"].str.strip() != ""]
    df = df[df["directions"].str.strip() != ""]

    if "Unnamed: 0" in df.columns:
        df["csv_row_id"] = df["Unnamed: 0"].astype(int)
    else:
        df["csv_row_id"] = df.index.astype(int)

    documents = [row_to_document(row) for _, row in df.iterrows()]
    print(f"Documents: {len(documents):,}")
    print(f"Sample metadata: {documents[0].metadata}")

    if CHROMA_DIR.exists():
        print(f"Removing old {CHROMA_DIR}...")
        shutil.rmtree(CHROMA_DIR)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = None
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=str(CHROMA_DIR),
            )
        else:
            vectorstore.add_documents(batch)
        print(f"Indexed {min(i + BATCH_SIZE, len(documents)):,} / {len(documents):,}")

    # Verify
    sample = vectorstore.similarity_search("chicken", k=1)[0]
    assert "csv_row_id" in sample.metadata, "csv_row_id missing after ingest!"
    print(f"\nDone. chroma_db at {CHROMA_DIR}")
    print(f"Verify: csv_row_id={sample.metadata['csv_row_id']} | {sample.metadata['recipe_name']}")


if __name__ == "__main__":
    main()
