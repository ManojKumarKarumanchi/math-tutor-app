"""
Explainer Agent: Generates student-friendly explanations of solutions.
Uses model fallback on rate limits.
"""

from typing import Any

from agno.agent import Agent


def create_explainer_agent(context_docs: list = None, model_obj: Any = None) -> Agent:
    """Create and return the Explainer Agent. model_obj: optional model object override."""
    if model_obj is None:
        from config.models import get_agent_model

        model_obj = get_agent_model()

    context = ""
    if context_docs:
        context = "\n\n**Reference Materials:**\n" + "\n".join(
            [f"- {doc[:150]}..." for doc in context_docs[:2]]
        )

    instructions = f"""
        You are a **Math Tutor for a -level student**.
        IMPORTANT:
        * The **solution steps are already provided**.
        * Your job is **NOT** to repeat, re-derive, or list steps again.
        Your role is to provide **conceptual and intuitive explanation** only.
        Focus on the following:
        1. REASONING (the “why”):
        * Why is this approach appropriate for this problem?
        * Why does differentiating help in finding stationary points?
        * What is the intuition behind setting the derivative equal to zero?

        2. KEY CONCEPTS:
        * What core ideas are being tested (e.g., derivative as slope, turning points)?
        * Which standard  concepts or formulas are involved, and why they matter?

        3. COMMON MISTAKES:
        * Typical errors students make in similar problems
        * Conceptual misunderstandings to watch out for

        4. LEARNING TAKEAWAYS:
        * What pattern or method should the student remember?
        * How this idea connects to maxima–minima, graph analysis, or optimization

        CONSTRAINTS:
        * Do NOT restate the solution steps.
        * Do NOT perform calculations.
        * Keep the explanation concise and clear: 2-3 sentences maximum.
        * Prioritize intuition, clarity, and exam relevance.
        {context}
"""

    return Agent(
        name="Explainer Agent",
        model=model_obj,
        instructions=instructions,
        markdown=True,
    )


def explain_solution(
    solution: str,
    context_docs: list = None,
    parsed: dict = None,
    selected_model: str = None,
) -> str:
    """
    Generate a student-friendly explanation. Returns explanation string.
    selected_model: optional user-selected model ID.
    """
    prompt = f"Explain this solution in a student-friendly way:\n\n{solution}"
    if parsed:
        prompt = f"Problem: {parsed.get('problem_text', '')}\n\n{prompt}"

    def explain_with_agent(agent: Agent):
        return agent.run(prompt).content

    try:
        from config.models import get_agent_model

        preferred_model_obj = get_agent_model(selected=selected_model)
        # Use selected model only (no fallback)
        agent = create_explainer_agent(context_docs, preferred_model_obj)
        return explain_with_agent(agent)
    except Exception as e:
        print(f"Explainer error (all models failed): {e}")
        return f"Error generating explanation: {e}"
