"""Utilidad de solo-lectura para construir el árbol de categorías del sidebar.

Lee `data/recipes_spanish.csv` (o `recipes.csv` como respaldo), parsea `cuisine_path`
y devuelve un árbol de 3 niveles con etiquetas en español:

    Categoría -> Subcategoría -> [Receta (nombre + imagen), ...]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

MAX_SUBCATEGORIES_PER_ROOT = 12
MAX_RECIPES_PER_SUBCATEGORY = 12

DEFAULT_RECIPES_CSV = Path(__file__).resolve().parent.parent / "data" / "recipes_spanish.csv"
FALLBACK_RECIPES_CSV = Path(__file__).resolve().parent.parent / "data" / "recipes.csv"
LABELS_JSON_PATH = Path(__file__).resolve().parent / "cuisine_labels_es.json"

# Etiquetas frecuentes de cuisine_path (AllRecipes) -> español
CUISINE_LABELS_ES: dict[str, str] = {
    "Appetizers and Snacks": "Aperitivos y tentempiés",
    "Aptizers and Snacks": "Aperitivos y tentempiés",
    "Apetizers and Snacks": "Aperitivos y tentempiés",
    "BBQ & Grilling": "Parrilla y barbacoa",
    'BBQ &quot; Grilling': "Parrilla y barbacoa",
    "Bread": "Pan",
    "Recetas de pan": "Pan",
    "Breakfast and Brunch": "Desayuno y brunch",
    "Breakfast y Brunch": "Desayuno y brunch",
    "Cuisine": "Cocinas del mundo",
    "Desserts": "Postres",
    "Drinks Recipes": "Bebidas",
    "Bebidas Recetas": "Bebidas",
    "Bebidas Recetas ": "Bebidas",
    "Antipasto Recipes": "Antipastos",
    "Beans and Peas": "Legumbres",
    "Bruschetta Recipes": "Bruschettas",
    "Canapes and Crostini Recipes": "Canapés y crostinis",
    "Meat Appetizers": "Aperitivos con carne",
    "Everyday Cooking": "Cocina diaria",
    "Cada día Cocina": "Cocina diaria",
    "Cocina diaria ": "Cocina diaria",
    "Fruits and Vegetables": "Frutas y verduras",
    " Frutas y verduras": "Frutas y verduras",
    "Holidays and Events Recipes": "Fiestas y celebraciones",
    "Main Dishes": "Platos principales",
    "Meat and Poultry": "Carne y aves",
    "Carne y aves": "Carne y aves",
    "Mexican": "Mexicana",
    "Quick Bread Recipes": "Pan rápido",
    " Pan": "Pan",
    "Salad": "Ensaladas",
    "Sauces and Condiments": "Salsas y condimentos",
    "Aceites y Condimentos": "Salsas y condimentos",
    "Seafood": "Mariscos y pescados",
    "Side Dish": "Guarniciones",
    "Soup Recipes": "Sopas",
    " Recetas de sopa ": "Sopas",
    "Soups, Stews and Chili Recipes": "Sopas, guisos y chilis",
    "Trusted Brands: Recipes and Tips": "Marcas y consejos",
    "Asian": "Asiática",
    "Cakes": "Tortas",
    "Chicken": "Pollo",
    "Cocktail Recipes": "Cócteles",
    "Cookies": "Galletas",
    "Crisps and Crumbles Recipes": "Crumble y crocantes",
    "Dips and Spreads Recipes": "Dips y untables",
    "Drinks": "Bebidas",
    "European": "Europea",
    "Fruit Desserts": "Postres con fruta",
    "Fruit Salad Recipes": "Ensaladas de fruta",
    "Green Salad Recipes": "Ensaladas verdes",
    "Latin American": "Latinoamericana",
    "Pies": "Tartas y pasteles",
    "Pork": "Cerdo",
    "Quick Bread Recipes": "Pan rápido",
    "Sauces and Condiments": "Salsas y condimentos",
    "Smoothie Recipes": "Smoothies",
    "Soup Recipes": "Sopas",
    "Specialty Dessert Recipes": "Postres especiales",
    "Apple Dessert Recipes": "Postres con manzana",
    "Apple Pie Recipes": "Tartas de manzana",
    "Apple Crisps and Crumbles Recipes": "Crumble de manzana",
    "Applesauce Recipes": "Compotas de manzana",
    " Recetas de la plancha ": "A la plancha",
    "Cobblers": "Cobblers de fruta",
    "Custards and Puddings": "Natillas y pudines",
    "Fillings": "Rellenos",
    "Frostings and Icings": "Glaseados y coberturas",
    "Frozen Dessert Recipes": "Postres helados",
    "Candy Recipes": "Dulces",
    "Brownie Recipes": "Brownies",
    "Cheesecake Recipes": "Cheesecakes",
    "Chocolate Dessert Recipes": "Postres de chocolate",
    "Cupcake Recipes": "Cupcakes",
    "Ice Cream Recipes": "Helados",
    "Vegetable Salad Recipes": "Ensaladas de verduras",
    "Pasta Salad Recipes": "Ensaladas de pasta",
    "Potato Salad Recipes": "Ensaladas de papa",
    "Coleslaw Recipes": "Ensaladas de col",
}

_WORD_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (" Recipes", ""),
    (" recipes", ""),
    ("Desserts", "Postres"),
    ("Dessert", "Postre"),
    ("Salad", "Ensalada"),
    ("Salads", "Ensaladas"),
    ("Chicken", "Pollo"),
    ("Pork", "Cerdo"),
    ("Beef", "Carne"),
    ("Soup", "Sopa"),
    ("Soups", "Sopas"),
    ("Cocktail", "Cóctel"),
    ("Cookies", "Galletas"),
    ("Cakes", "Tortas"),
    ("Pies", "Tartas"),
    ("Bread", "Pan"),
    ("Drinks", "Bebidas"),
    ("Fruit", "Fruta"),
    ("Apple", "Manzana"),
    ("Smoothie", "Smoothie"),
    ("European", "Europea"),
    ("Asian", "Asiática"),
    ("Latin American", "Latinoamericana"),
)


def _merged_labels() -> dict[str, str]:
    labels: dict[str, str] = {}
    if LABELS_JSON_PATH.exists():
        try:
            loaded = json.loads(LABELS_JSON_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                labels.update(loaded)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"⚠️ No se pudo cargar {LABELS_JSON_PATH}: {exc}")
    # Las entradas manuales tienen prioridad sobre el JSON autogenerado.
    labels.update(CUISINE_LABELS_ES)
    return labels


@dataclass(frozen=True)
class RecipeEntry:
    """Una receta individual lista para mostrarse en el sidebar."""

    name: str
    image_url: str | None


CuisineTree = dict[str, dict[str, list[RecipeEntry]]]


def resolve_recipes_csv_path(csv_path: Path | None = None) -> Path:
    if csv_path is not None:
        return csv_path
    if DEFAULT_RECIPES_CSV.exists():
        return DEFAULT_RECIPES_CSV
    return FALLBACK_RECIPES_CSV


def _split_path(raw_path: str) -> list[str]:
    if not raw_path or not isinstance(raw_path, str):
        return []
    return [segment.strip() for segment in raw_path.strip("/").split("/") if segment.strip()]


def label_cuisine_es(segment: str, labels: dict[str, str] | None = None) -> str:
    """Traduce un segmento de cuisine_path al español para mostrar en la UI."""
    cleaned = re.sub(r"\s+", " ", segment.strip())
    if not cleaned:
        return cleaned

    label_map = labels if labels is not None else _merged_labels()

    if cleaned in label_map:
        return label_map[cleaned]

    lowered_map = {key.lower(): value for key, value in label_map.items()}
    hit = lowered_map.get(cleaned.lower())
    if hit:
        return hit

    text = cleaned
    for source, target in _WORD_REPLACEMENTS:
        text = text.replace(source, target)

    text = re.sub(r"\band\b", "y", text, flags=re.IGNORECASE)
    text = re.sub(r"\bRecetas\b", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text or cleaned


_label_es = label_cuisine_es


def _csv_columns(path: Path) -> list[str]:
    import pandas as pd

    return list(pd.read_csv(path, nrows=0).columns)


def _recipe_id_column(df) -> None:
    if "Unnamed: 0" in df.columns:
        df["id"] = df["Unnamed: 0"].astype(int)
    else:
        df["id"] = df.index.astype(int)


def _load_sidebar_dataframe(spanish_path: Path, english_path: Path):
    import pandas as pd

    spanish_cols = ["recipe_name", "img_src"]
    if "Unnamed: 0" in _csv_columns(spanish_path):
        spanish_cols = ["Unnamed: 0", *spanish_cols]
    df = pd.read_csv(spanish_path, usecols=spanish_cols)
    _recipe_id_column(df)

    path_source = english_path if english_path.exists() else spanish_path
    path_cols = ["cuisine_path"]
    if "Unnamed: 0" in _csv_columns(path_source):
        path_cols = ["Unnamed: 0", "cuisine_path"]
    paths = pd.read_csv(path_source, usecols=path_cols)
    _recipe_id_column(paths)
    merged = df.merge(paths[["id", "cuisine_path"]], on="id", how="left")
    return merged


def build_cuisine_tree(csv_path: Path | None = None) -> CuisineTree:
    """Arma categoría -> subcategoría -> recetas con nombres y etiquetas en español."""
    spanish_path = resolve_recipes_csv_path(csv_path)
    english_path = FALLBACK_RECIPES_CSV
    raw_tree: dict[str, dict[str, list[RecipeEntry]]] = {}

    try:
        if not spanish_path.exists():
            return {}

        label_map = _merged_labels()
        df = _load_sidebar_dataframe(spanish_path, english_path)

        for _, row in df.iterrows():
            raw_path = row.get("cuisine_path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue

            parts = _split_path(raw_path)
            if len(parts) < 2:
                continue

            root = _label_es(parts[0], label_map)
            subcat = _label_es(parts[1], label_map)
            name = row.get("recipe_name")
            if not isinstance(name, str) or not name.strip():
                continue

            image_url = row.get("img_src")
            image_url = image_url if isinstance(image_url, str) and image_url.strip() else None

            raw_tree.setdefault(root, {}).setdefault(subcat, [])
            raw_tree[root][subcat].append(RecipeEntry(name=name.strip(), image_url=image_url))

    except Exception as exc:
        print(f"⚠️ No se pudo construir el árbol de categorías desde {spanish_path}: {exc}")
        return {}

    ordered_tree: CuisineTree = {}
    for root in sorted(raw_tree.keys(), key=lambda value: value.casefold()):
        subcats = raw_tree[root]
        ordered_subcats: dict[str, list[RecipeEntry]] = {}

        subcat_names = sorted(
            subcats.keys(),
            key=lambda name: (-len(subcats[name]), name.casefold()),
        )
        if MAX_SUBCATEGORIES_PER_ROOT is not None:
            subcat_names = subcat_names[:MAX_SUBCATEGORIES_PER_ROOT]

        for subcat in subcat_names:
            recipes = subcats[subcat]
            seen: dict[str, RecipeEntry] = {}
            for recipe in sorted(recipes, key=lambda recipe: recipe.name.casefold()):
                seen.setdefault(recipe.name.casefold(), recipe)

            recipe_list = list(seen.values())
            if MAX_RECIPES_PER_SUBCATEGORY is not None:
                recipe_list = recipe_list[:MAX_RECIPES_PER_SUBCATEGORY]

            ordered_subcats[subcat] = recipe_list

        ordered_tree[root] = ordered_subcats

    return ordered_tree
