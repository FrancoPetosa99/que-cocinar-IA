"""Recipe-focused agent tools."""

from __future__ import annotations

from langchain_core.tools import tool

from backend.config import get_llm
from backend.database.read_only_interfaces import search_recipes

SCALING_PROMPT = """You are a recipe scaling expert. You receive a recipe and must adapt it from {current} to {target} servings.

Rules:
- Multiply every numeric quantity by factor {factor:.4f}.
- Keep original units.
- If an ingredient has no numeric amount, use "to taste" or "proportional amount".
- At the end, include a brief estimated macro summary if nutritional data is available.

Original recipe:
{recipe}

Respond in English only.
"""

SUBSTITUTION_PROMPT = """You are a culinary substitution expert.

Ingredient to replace: {ingredient}
Dietary constraint (if any): {constraint}

Suggest 2-3 practical substitutes with:
- substitute name
- replacement ratio (e.g. 1:1, 3/4 cup per cup)
- brief note on flavor or texture impact

Respond in English only. Be concise and actionable.
"""


@tool
def recipe_retriever(
    query: str,
    max_total_time_min: int | None = None,
    min_protein_g: float | None = None,
    max_calories: float | None = None,
) -> str:
    """
    Search the recipe database by ingredients, theme, or constraints.

    Args:
        query: What the user is looking for (ingredients, weather, style). Use English.
        max_total_time_min: Maximum time in minutes (for quick recipes).
        min_protein_g: Minimum protein in grams (high-protein diets).
        max_calories: Maximum calories per serving.
    """
    return search_recipes(
        query,
        max_total_time=max_total_time_min,
        min_protein=min_protein_g,
        max_calories=max_calories,
    )


@tool
def scaling_expert(
    recipe_text: str,
    current_servings: int,
    target_servings: int,
) -> str:
    """
    Scale recipe quantities from N to M servings.

    Args:
        recipe_text: Full recipe text to scale (English).
        current_servings: Current number of servings.
        target_servings: Desired number of servings.
    """
    if current_servings <= 0 or target_servings <= 0:
        return "Servings must be positive numbers."

    factor = target_servings / current_servings
    llm = get_llm(streaming=False)
    prompt = SCALING_PROMPT.format(
        current=current_servings,
        target=target_servings,
        factor=factor,
        recipe=recipe_text,
    )
    response = llm.invoke(prompt)
    return response.content


@tool
def substitution_expert(
    ingredient: str,
    dietary_constraint: str = "",
) -> str:
    """
    Suggest alternatives for a missing ingredient or dietary constraint.

    Args:
        ingredient: Missing or replaceable ingredient (English).
        dietary_constraint: Optional constraint (vegan, gluten-free, etc.).
    """
    llm = get_llm(streaming=False)
    prompt = SUBSTITUTION_PROMPT.format(
        ingredient=ingredient,
        constraint=dietary_constraint or "none",
    )
    response = llm.invoke(prompt)
    return response.content


TOOLS = [recipe_retriever, scaling_expert, substitution_expert]
