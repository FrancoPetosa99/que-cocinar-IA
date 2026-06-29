from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from backend.config import get_llm

class QueryClassification(BaseModel):
    valid: bool

CLASSIFIER_PROMPT = """
You are a domain classifier for Qué Cocinar IA.

Your task is to determine whether the user's request belongs to the cooking domain.

A VALID request includes:

- recipes
- ingredients
- meals
- cooking techniques
- nutrition related to recipes
- recipe scaling
- ingredient substitutions
- meal planning
- baking
- grilling
- frying
- kitchen utensils
- food preparation

An INVALID request is anything unrelated to cooking.

Return ONLY a valid JSON object.

Example:
{{
    "valid": true
}}

Rules:
- Do not include explanations.
- Do not use markdown.
- Do not wrap the JSON in ```.

User message:

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