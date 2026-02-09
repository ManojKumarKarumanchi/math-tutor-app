"""
Parser Agent: Converts raw text into structured problem representation.
Detects ambiguity and triggers HITL when needed. Uses model fallback on rate limits.
"""

import json
from typing import Any

from agno.agent import Agent


def create_parser_agent(model_obj: Any = None) -> Agent:
    """Create and return the Parser Agent. model_obj: optional model object override."""
    if model_obj is None:
        from config.models import get_agent_model

        model_obj = get_agent_model()

    instructions = """You are a Math Parser Agent for JEE-level problems. Extract structured information and NORMALIZE the problem text. Be LENIENT: only flag ambiguity when the problem is truly unsolvable.

**Normalize in problem_text:**
- "r of x", "f of x" → r(x), f(x)
- "whole ^2", "whole squared", "whole **2" → squared, e.g. (expression)^2
- "(x+1)(x-2)" or "x+1 * x-2" (when clearly product) → (x+1)(x-2)
- Fix obvious OCR/ASR typos if the intent is clear (e.g. "increasing and decreasing" = find intervals where increasing/decreasing)

**needs_clarification**: Set to true ONLY when:
- Critical information is missing and cannot be inferred (e.g. "Find the derivative" with no function at all)
- The problem is not clearly a math problem
- Two or more completely different interpretations are equally likely

**Do NOT set needs_clarification for:**
- Grammatical or spelling errors if the math intent is clear
- Informal wording ("r of x", "whole square", "find where it is increasing or decreasing")
- Minor ambiguities that have a standard interpretation (e.g. "increasing and decreasing" → find intervals of increase/decrease)
- OCR/ASR noise if the problem is identifiable (e.g. "Determine the function r of x = to x + 1 * x - 2 whole ^2 is increasing and decreasing" → interpret as r(x) = (x+1)(x-2)^2, find where increasing/decreasing)

Extract:
1. **topic**: algebra, calculus, probability, linear_algebra, other
2. **variables**: e.g. ["x", "y"]
3. **constraints**: e.g. ["x > 0"]
4. **problem_text**: CLEANED and NORMALIZED statement (use standard math notation)
5. **needs_clarification**: true only if truly unsolvable without more info
6. **clarification_reason**: only if needs_clarification is true

Return ONLY a valid JSON object. No markdown, no explanation.

Example: "Determine the function r of x = to x + 1 * x - 2 whole ^2 is increasing and decreasing"
{
  "topic": "calculus",
  "variables": ["x"],
  "constraints": [],
  "problem_text": "Determine where the function r(x) = (x+1)(x-2)^2 is increasing and decreasing",
  "needs_clarification": false,
  "clarification_reason": ""
}

Example: "Find the derivative"
{
  "topic": "calculus",
  "variables": [],
  "constraints": [],
  "problem_text": "Find the derivative",
  "needs_clarification": true,
  "clarification_reason": "No function specified to differentiate"
}
"""

    return Agent(
        name="Parser Agent",
        model=model_obj,
        instructions=instructions,
        markdown=False,
    )


def parse_problem(text: str, selected_model: str = None) -> dict:
    """
    Parse a math problem into structured format with automatic model fallback.
    selected_model: optional user-selected model ID.
    """

    def parse_with_agent(agent: Agent) -> dict:
        response = agent.run(text)
        content = response.content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            if "```json" in content:
                part = content.split("```json")[1].split("```")[0].strip()
                return json.loads(part)
            if "```" in content:
                part = content.split("```")[1].split("```")[0].strip()
                return json.loads(part)
            return {
                "topic": "other",
                "variables": [],
                "constraints": [],
                "problem_text": text,
                "needs_clarification": False,
                "clarification_reason": "",
            }

    try:
        from config.models import get_agent_model

        preferred_model_obj = get_agent_model(selected=selected_model)
        # Use selected model only (no fallback)
        agent = create_parser_agent(preferred_model_obj)
        return parse_with_agent(agent)

    except Exception as e:
        print(f"Parser error (all models failed): {e}")
        error_str = str(e)
        if "rate limit" in error_str.lower() or "rate_limit" in error_str.lower():
            return {
                "error": {
                    "message": error_str,
                    "type": "rate_limit",
                    "code": "rate_limit_exceeded",
                }
            }
        return {
            "topic": "other",
            "variables": [],
            "constraints": [],
            "problem_text": text,
            "needs_clarification": False,
            "clarification_reason": "",
            "error": str(e),
        }
