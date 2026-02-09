"""
Model fallback: retry agent calls with fallback models when rate limits hit.
Supports Groq and Gemini models.
"""

import time
from typing import Any, Callable, Optional

from config.models import get_fallback_models, is_rate_limit_error


def call_with_fallback(
    create_agent_fn: Callable[[str], Any],
    agent_call_fn: Callable[[Any], Any],
    agent_name: str = "Agent",
    max_retries: int = 3,
    preferred_model: str = None,
) -> Any:
    """
    Call an agent with automatic fallback to other models on rate limits.

    create_agent_fn: takes model_string, returns Agent.
    agent_call_fn: takes agent, returns result (e.g. lambda a: a.run(prompt)).
    agent_name: for logging.
    max_retries: per model.
    preferred_model: user-selected model to try first.

    Returns result from agent_call_fn. Raises last exception if all fail.
    """
    # If user selected a specific model, use ONLY that model (no fallback)
    if preferred_model:
        models = [preferred_model]
        print(f"{agent_name}: Using selected model: {preferred_model}")
    else:
        models = get_fallback_models()

    last_error = None

    for i, model in enumerate(models):
        for attempt in range(max_retries):
            try:
                agent = create_agent_fn(model)
                result = agent_call_fn(agent)
                if i > 0 or attempt > 0:
                    print(f"{agent_name}: Success with {model}")
                return result
            except Exception as e:
                last_error = e
                error_str = str(e)

                # Log the error for debugging
                print(f"{agent_name}: Error with {model}: {str(e)[:100]}")

                if is_rate_limit_error(e):
                    # For rate limits, try next model or retry
                    if i < len(models) - 1 and attempt == 0:
                        print(
                            f"{agent_name}: Rate limit on {model}, trying next model..."
                        )
                        break
                    if attempt < max_retries - 1:
                        wait_time = 2**attempt
                        print(f"{agent_name}: Retrying {model} after {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    if i < len(models) - 1:
                        break
                else:
                    # For non-rate-limit errors, try next model
                    if i < len(models) - 1:
                        print(
                            f"{agent_name}: Non-rate-limit error on {model}, trying next model..."
                        )
                        break
                    raise

    if last_error:
        raise last_error
    raise Exception(f"{agent_name}: All models exhausted")


def create_agent_with_fallback(
    agent_creator: Callable[[str], Any],
    prompt: str,
    agent_name: str = "Agent",
) -> Any:
    """
    Create agent and run prompt with automatic fallback.
    Returns agent response.
    """
    return call_with_fallback(
        create_agent_fn=agent_creator,
        agent_call_fn=lambda agent: agent.run(prompt),
        agent_name=agent_name,
    )
