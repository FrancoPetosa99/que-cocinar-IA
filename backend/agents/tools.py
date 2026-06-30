from __future__ import annotations

from langchain_core.tools import tool

from backend.agents.scaling_agent import build_scaling_agent
from backend.agents.substitution_agent import build_substitution_agent

@tool
def find_relevant_recipes(query: str) -> str:
    """
    Recupera las recetas más relevantes para una consulta del usuario.

    Utiliza búsqueda semántica sobre Chroma y luego obtiene la información
    completa desde SQLite.

    Utilizá esta herramienta cuando el usuario:

    - pregunte qué cocinar;
    - mencione uno o más ingredientes;
    - solicite una receta;
    - busque recetas rápidas, saludables o con restricciones nutricionales.

    Nunca inventa recetas; siempre consulta la base de datos.

    Args:
        query: Consulta del usuario en español.

    Returns:
        Texto con hasta cuatro recetas candidatas.
    """

    recipe_ids = search_recipe_ids(query=query, k=4)

    if not recipe_ids:
        return "No se encontraron recetas."

    recipes = []

    for recipe_id in recipe_ids:
        recipe = get_recipe_by_id(recipe_id)

        if recipe is None:
            continue

        recipes.append(
            f"""
                ID:             {recipe.id}
                Nombre:         {recipe.recipe_name}
                Ingredientes:   {recipe.ingredients}
                Preparación:    {recipe.directions}
                Porciones: {recipe.servings}
                Calorías: {recipe.calories}
                Proteínas: {recipe.protein}
                Tiempo total: {recipe.total_time} minutos
                Fuente verificada:
                csv_row_id={recipe.id} | name={recipe.recipe_name}
            """.strip()
        )
    return "\n\n-----------------------------\n\n".join(recipes)

@tool
def scale_recipe(recipe_text: str, current_servings: int, target_servings: int) -> str:
    """
    Adapta una receta existente a una nueva cantidad de porciones.

    Utilizá esta herramienta cuando el usuario quiera:

    - cocinar para más o menos personas;
    - aumentar o reducir las porciones de una receta;
    - escalar las cantidades de los ingredientes;
    - ajustar una receta a una cantidad diferente de comensales.

    Args:
        recipe_text: Texto completo de la receta que debe adaptarse.
        current_servings: Cantidad de porciones que produce actualmente la receta.
        target_servings: Cantidad de porciones deseada.

    Returns:
        La receta adaptada con las nuevas cantidades de ingredientes y,
        cuando corresponda, un resumen nutricional actualizado.
    """

    agent = build_scaling_agent()

    result = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"""
                        Receta:
                        {recipe_text}
                        Porciones actuales: {current_servings}
                        Porciones deseadas: {target_servings}
                    """
                )
            ]
        }
    )

    return result["messages"][-1].content

@tool
def substitute_ingredient(ingredient: str, dietary_constraint: str = "") -> str:
    """
    Adapta una receta a una nueva cantidad de porciones.

    Utilizá esta herramienta cuando el usuario quiera:

    - aumentar o reducir porciones;
    - adaptar cantidades;
    - cocinar para más o menos personas.

    Requiere la receta original.

     Args:
        ingredient: Ingrediente que el usuario desea reemplazar.
        dietary_constraint: Restricción alimentaria opcional
            (por ejemplo: vegano, vegetariano, sin gluten, sin lactosa).

    Returns:
        Una lista de sustitutos prácticos indicando, cuando sea posible,
        la proporción de reemplazo y el impacto esperado en el sabor o la textura.
        
    """

    agent = build_substitution_agent()

    result = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"""
                    Ingrediente:
                    {ingredient}
                    Restricción:
                    {dietary_constraint}
                    """
                )
            ]
        }
    )

    return result["messages"][-1].content