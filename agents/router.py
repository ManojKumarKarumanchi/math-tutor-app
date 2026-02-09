"""
Intent Router Agent: Classifies problem type and determines solution strategy.
"""

from agno.agent import Agent


def create_router_agent(selected_model: str = None) -> Agent:
    """
    Create and return the Router Agent.
    selected_model: optional user-selected model ID.
    """
    from config.models import get_agent_model

    model_obj = get_agent_model(selected=selected_model)

    instructions = """You are an Intent Router Agent for math problems.

Given a parsed problem structure, determine:
1. The primary math topic
2. Whether symbolic math tools are needed
3. The solution strategy

Return ONLY the topic name: algebra, calculus, probability, linear_algebra, or other
"""

    return Agent(
        name="Router Agent",
        model=model_obj,
        instructions=instructions,
        markdown=False,
    )


def route_topic(parsed: dict, selected_model: str = None) -> str:
    """
    Route problem to topic. Returns algebra, calculus, probability, linear_algebra, or other.
    selected_model: optional user-selected model ID (for consistency, though topic is extracted from parsed dict).
    """
    topic = parsed.get("topic", "other")
    valid = ["algebra", "calculus", "probability", "linear_algebra", "other"]
    if topic not in valid:
        topic = "other"
    return topic
