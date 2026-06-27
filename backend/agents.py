"""LangGraph agent, tools, and session memory (internal prompts in English)."""

from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from backend.config import get_llm
from backend.database import search_recipes

SYSTEM_PROMPT = """You are Qué Cocinar IA, an assistant specialized exclusively in cooking recipes.

OBJECTIVE
Help the user cook using available ingredients and answer only queries related to cooking, recipes, ingredients, and culinary techniques.

AVAILABLE TOOLS
- recipe_retriever: search real recipes in the database. Use for available ingredients, themes, weather (hot/cold), time or macro constraints.
- scaling_expert: scale ingredient quantities when the user asks for more or fewer servings.
- substitution_expert: suggest replacements when an ingredient is missing or there are dietary restrictions.

RESPONSE RULES
1. Suggest ONE recipe per response.
2. Pick the best recipe for the user's ingredients or constraints.
3. Do not list multiple alternatives unless explicitly requested.
4. No unnecessary meta-commentary ("this is very popular...", "ideal for gaining muscle...").
5. Be direct and action-oriented.
6. If ingredients are missing, use substitution_expert or suggest simple adaptations.
7. If the query is not about cooking, respond: "I can only help with recipes and cooking-related topics."
8. For quick recipes, use recipe_retriever with a low max_total_time_min (e.g. 20).
9. For athletes or high protein, use recipe_retriever with a high min_protein_g (e.g. 30).
10. ALWAYS use recipe_retriever before suggesting a recipe. NEVER invent recipes or names.
11. Pick ONE recipe from retriever results and base your answer on that row.
12. At the END of EVERY response include exactly this line (do not change the format):
    Verified source: csv_row_id=XXX | name=YYY
    where XXX is the csv_row_id of the chosen recipe and YYY is its exact recipe_name from the retriever.

RESPONSE FORMAT
Recipe name.

Ingredients:
* brief ingredient list

Directions:
1. Step 1
2. Step 2
3. Step 3

Keep the response brief and practical. Respond in English (translation to Spanish is handled downstream).
"""

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

_checkpointer = MemorySaver()
_agent = None


def build_agent():
    """Create (or return cached) LangGraph ReAct agent with session memory."""
    global _agent
    if _agent is not None:
        return _agent

    llm = get_llm(streaming=True)
    _agent = create_react_agent(
        llm,
        TOOLS,
        checkpointer=_checkpointer,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )
    return _agent


def get_agent_config(thread_id: str) -> dict:
    """Return LangGraph config for a given session thread."""
    return {"configurable": {"thread_id": thread_id}}
