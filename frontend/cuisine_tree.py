"""Utilidad de solo-lectura para construir el árbol de categorías del sidebar.

Este módulo es exclusivamente de UI: lee `data/recipes.csv`, parsea la columna
`cuisine_path` (formato `/Categoria/Subcategoria/.../`) y devuelve una
estructura de árbol de 3 niveles para renderizar en el sidebar del chat:

    Categoría -> Subcategoría -> [Receta (nombre + imagen), ...]

No depende de `backend/` ni modifica ningún dato: es una vista de solo
lectura sobre el CSV, pensada únicamente para alimentar la UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Tope de subcategorías que se muestran por categoría principal, para que el
# acordeón no quede gigantesco. None = sin límite.
MAX_SUBCATEGORIES_PER_ROOT = 12

# Tope de recetas que se muestran por subcategoría (algunas subcategorías
# tienen 100+ recetas; mostrar todas haría el sidebar inmanejable).
MAX_RECIPES_PER_SUBCATEGORY = 12


@dataclass(frozen=True)
class RecipeEntry:
    """Una receta individual lista para mostrarse en el sidebar."""

    name: str
    image_url: str | None


# Categoría -> Subcategoría -> lista de RecipeEntry
CuisineTree = dict[str, dict[str, list[RecipeEntry]]]


def _split_path(raw_path: str) -> list[str]:
    """Convierte '/Desserts/Pies/Apple Pie Recipes/' en ['Desserts', 'Pies', 'Apple Pie Recipes']."""
    if not raw_path or not isinstance(raw_path, str):
        return []
    return [segment.strip() for segment in raw_path.strip("/").split("/") if segment.strip()]


def build_cuisine_tree(csv_path: Path) -> CuisineTree:
    """Lee recipes.csv y arma un árbol de 3 niveles: categoría -> subcategoría -> recetas.

    Devuelve un dict ordenado alfabéticamente:
        {
            "Desserts": {
                "Pies": [RecipeEntry(name="Apple Pie", image_url="https://..."), ...],
                ...
            },
            ...
        }

    Si el CSV no existe o falla la lectura, devuelve un dict vacío en lugar
    de lanzar una excepción, para que el sidebar simplemente no se muestre
    sin romper el resto de la app.
    """
    # root -> subcat -> lista de (nombre, imagen), sin deduplicar todavía
    raw_tree: dict[str, dict[str, list[RecipeEntry]]] = {}

    try:
        import pandas as pd  # import local: evita exigir pandas si no se usa el sidebar

        if not csv_path.exists():
            return {}

        df = pd.read_csv(
            csv_path,
            usecols=["recipe_name", "cuisine_path", "img_src"],
        )

        for _, row in df.iterrows():
            parts = _split_path(row.get("cuisine_path"))
            if len(parts) < 2:
                # Sin subcategoría definida: no hay donde anclar la receta
                # en un árbol de 3 niveles, así que se omite del sidebar.
                continue

            root, subcat = parts[0], parts[1]
            name = row.get("recipe_name")
            if not isinstance(name, str) or not name.strip():
                continue

            image_url = row.get("img_src")
            image_url = image_url if isinstance(image_url, str) and image_url.strip() else None

            raw_tree.setdefault(root, {}).setdefault(subcat, [])
            raw_tree[root][subcat].append(RecipeEntry(name=name.strip(), image_url=image_url))

    except Exception as exc:
        print(f"⚠️ No se pudo construir el árbol de categorías desde {csv_path}: {exc}")
        return {}

    ordered_tree: CuisineTree = {}
    for root in sorted(raw_tree.keys()):
        subcats = raw_tree[root]
        ordered_subcats: dict[str, list[RecipeEntry]] = {}

        subcat_names = sorted(subcats.keys())
        if MAX_SUBCATEGORIES_PER_ROOT is not None:
            subcat_names = subcat_names[:MAX_SUBCATEGORIES_PER_ROOT]

        for subcat in subcat_names:
            recipes = subcats[subcat]
            # Recetas ordenadas alfabéticamente y deduplicadas por nombre,
            # conservando la primera imagen encontrada para cada una.
            seen: dict[str, RecipeEntry] = {}
            for recipe in sorted(recipes, key=lambda r: r.name):
                seen.setdefault(recipe.name, recipe)

            recipe_list = list(seen.values())
            if MAX_RECIPES_PER_SUBCATEGORY is not None:
                recipe_list = recipe_list[:MAX_RECIPES_PER_SUBCATEGORY]

            ordered_subcats[subcat] = recipe_list

        ordered_tree[root] = ordered_subcats

    return ordered_tree