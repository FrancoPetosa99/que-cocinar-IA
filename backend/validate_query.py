from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from backend.config import get_llm


class QueryClassification(BaseModel):
    valid: bool


CLASSIFIER_PROMPT = """
Sos un clasificador de dominio para Qué Cocinar IA.

Tu tarea es determinar si la consulta del usuario pertenece al dominio de la cocina.

Una consulta VÁLIDA incluye:

- recetas
- ingredientes
- comidas
- técnicas de cocina
- nutrición relacionada con recetas
- escalado de recetas
- sustituciones de ingredientes
- planificación de comidas
- horneado
- parrilla
- fritura
- utensilios de cocina
- preparación de alimentos

Una consulta INVÁLIDA es cualquier cosa no relacionada con la cocina.

Devolvé SOLO un objeto JSON válido.

Ejemplo:
{{
    "valid": true
}}

Reglas:
- No incluyas explicaciones.
- No uses markdown.
- No envuelvas el JSON en ```.

Mensaje del usuario:

{message}
"""


def _extract_json(text: str) -> dict:
    """
    Extract JSON from the model response.
    Removes markdown code fences if present.
    """
    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):]

    if text.startswith("```"):
        text = text[len("```"):]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    return json.loads(text)


def classify_query(message: str) -> QueryClassification:
    """
    Determine whether a user query belongs to the cooking domain.
    """

    llm = get_llm(streaming=False)

    prompt = CLASSIFIER_PROMPT.format(message=message)

    response = llm.invoke(prompt)

    try:
        data = _extract_json(response.content)
        return QueryClassification.model_validate(data)

    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Invalid classifier response:\n{response.content}") from e
