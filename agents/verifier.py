"""
Verifier Agent: Validates solution correctness and confidence.
Uses model fallback on rate limits.
"""

import json
from typing import Any

from agno.agent import Agent


def create_verifier_agent(model_obj: Any = None) -> Agent:
    """Create and return the Verifier Agent. model_obj: optional model object override."""
    if model_obj is None:
        from config.models import get_agent_model

        model_obj = get_agent_model()

    instructions = """You are a Math Verifier Agent. Verify correctness of math solutions.

Check: mathematical correctness, units/domains, edge cases, completeness.

Respond with ONLY a valid JSON object. No markdown, no code fences.
{"verdict": "correct" or "incorrect" or "uncertain", "confidence": 0.0 to 1.0, "reasoning": "Brief explanation", "issues": []}

Example: {"verdict": "correct", "confidence": 0.95, "reasoning": "Solution correctly applies the formula", "issues": []}
"""

    return Agent(
        name="Verifier Agent",
        model=model_obj,
        instructions=instructions,
        markdown=False,
    )


def _extract_json_from_text(text: str) -> dict | None:
    """Extract a JSON object from model output (handles markdown, extra text)."""
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for marker in ("```json", "```"):
        if marker in text:
            try:
                part = text.split(marker, 1)[1].split("```")[0].strip()
                return json.loads(part)
            except (IndexError, json.JSONDecodeError):
                continue
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def verify_solution(
    solution: str, parsed: dict = None, selected_model: str = None
) -> tuple:
    """
    Verify solution. Returns (verdict, confidence_score).
    selected_model: optional user-selected model ID.
    """
    prompt = f"Verify this solution:\n\n{solution}"
    if parsed:
        prompt = f"Problem: {parsed.get('problem_text', '')}\n\n{prompt}"

    def verify_with_agent(agent: Agent):
        response = agent.run(prompt)
        content = getattr(response, "content", None) or ""
        if not content.strip():
            return ("uncertain", 0.7)
        result = _extract_json_from_text(content)
        if result is None:
            return ("uncertain", 0.7)
        verdict = result.get("verdict", "uncertain")
        if verdict not in ("correct", "incorrect", "uncertain"):
            verdict = "uncertain"
        try:
            confidence = float(result.get("confidence", 0.7))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.7
        return (verdict, confidence)

    try:
        from config.models import get_agent_model

        preferred_model_obj = get_agent_model(selected=selected_model)
        # Use selected model only (no fallback)
        agent = create_verifier_agent(preferred_model_obj)
        return verify_with_agent(agent)
    except Exception as e:
        print(f"Verifier error (all models failed): {e}")
        return "uncertain", 0.7
