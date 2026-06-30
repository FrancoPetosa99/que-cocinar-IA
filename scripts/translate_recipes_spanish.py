"""
Translate data/recipes.csv to Spanish (offline, argostranslate).

Output: data/recipes_spanish.csv

Usage:
    python scripts/translate_recipes_spanish.py
    python scripts/translate_recipes_spanish.py --limit 10   # smoke test
    python scripts/translate_recipes_spanish.py --resume     # continue partial output
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data" / "recipes.csv"
OUTPUT_CSV = ROOT / "data" / "recipes_spanish.csv"

TEXT_COLUMNS = (
    "recipe_name",
    "prep_time",
    "cook_time",
    "total_time",
    "yield",
    "ingredients",
    "directions",
    "cuisine_path",
    "timing",
)

# argostranslate works best below ~4500 chars per call
MAX_CHUNK_CHARS = 4000


def ensure_en_es_translator():
    import argostranslate.package
    import argostranslate.translate

    def _get_translate_fn():
        installed = argostranslate.translate.get_installed_languages()
        english = next((lang for lang in installed if lang.code == "en"), None)
        spanish = next((lang for lang in installed if lang.code == "es"), None)
        if english is not None and spanish is not None:
            translation = english.get_translation(spanish)
            if translation is not None:
                return translation.translate
        return None

    translate_fn = _get_translate_fn()
    if translate_fn is not None:
        return translate_fn

    print("Installing argostranslate EN -> ES package...")
    argostranslate.package.update_package_index()
    packages = argostranslate.package.get_available_packages()
    en_es = next(
        p for p in packages if p.from_code == "en" and p.to_code == "es"
    )
    download_path = en_es.download()
    argostranslate.package.install_from_path(download_path)

    translate_fn = _get_translate_fn()
    if translate_fn is None:
        raise RuntimeError("Failed to load argostranslate EN -> ES translator.")
    return translate_fn


def _is_blank(value) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    return not str(value).strip()


def _split_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(paragraph) <= max_chars:
            current = paragraph
            continue

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        buffer = ""
        for sentence in sentences:
            piece = f"{buffer} {sentence}".strip() if buffer else sentence
            if len(piece) <= max_chars:
                buffer = piece
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = sentence
        if buffer:
            current = buffer

    if current:
        chunks.append(current)

    return chunks or [text]


def translate_text(translate_fn, text: str) -> str:
    if _is_blank(text):
        return "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text)

    text = str(text)
    parts = _split_chunks(text)
    return "\n\n".join(translate_fn(part) for part in parts)


def translate_row(translate_fn, row: pd.Series) -> pd.Series:
    translated = row.copy()
    for column in TEXT_COLUMNS:
        if column in translated:
            translated[column] = translate_text(translate_fn, row[column])
    return translated


def load_input_dataframe(limit: int | None) -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    if limit is not None:
        df = df.head(limit).copy()
    return df


def translate_recipes(
    *,
    limit: int | None = None,
    resume: bool = False,
) -> Path:
    df = load_input_dataframe(limit)
    translate_fn = ensure_en_es_translator()

    start_index = 0
    rows: list[pd.Series] = []

    if resume and OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        rows = [existing.iloc[i] for i in range(len(existing))]
        start_index = len(existing)
        print(f"Resuming from row {start_index}/{len(df)}")

    for index in tqdm(range(start_index, len(df)), desc="Translating recipes"):
        rows.append(translate_row(translate_fn, df.iloc[index]))

        if (index + 1) % 25 == 0 or index + 1 == len(df):
            pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(rows):,} recipes to {OUTPUT_CSV}")
    return OUTPUT_CSV


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate recipes.csv to Spanish.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Translate only the first N rows (for testing).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from existing recipes_spanish.csv",
    )
    args = parser.parse_args()

    translate_recipes(limit=args.limit, resume=args.resume)


if __name__ == "__main__":
    main()
