"""Shared parsing helpers for recipe CSV / SQLite rows."""

from __future__ import annotations

import ast
import re

import pandas as pd


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


def parse_nutrition(value) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "calories": None,
        "protein_g": None,
        "carbs_g": None,
        "fat_g": None,
    }
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
