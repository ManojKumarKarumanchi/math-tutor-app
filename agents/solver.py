"""
Solver Agent: Generates step-by-step solutions using RAG and SymPy tools.
Uses model fallback on rate limits.
"""

from pathlib import Path
from typing import Any

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.tools import tool
from sympy import Symbol, diff, integrate, simplify, solve
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_solver_db = None


def _get_solver_db():
    """Get or create the shared SQLite database for solver agent history."""
    global _solver_db
    if _solver_db is None:
        db_path = Path(__file__).parent.parent / "tmp" / "solver_history.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _solver_db = SqliteDb(db_file=str(db_path))
    return _solver_db


@tool
def solve_equation(equation: str, variable: str = "x") -> str:
    """Solve an equation symbolically using SymPy. Returns solution as string."""
    try:
        var = Symbol(variable)
        expr = parse_expr(
            equation,
            transformations=standard_transformations
            + (implicit_multiplication_application,),
        )
        solutions = solve(expr, var)
        return f"Solutions: {solutions}"
    except Exception as e:
        return f"Error solving equation: {e}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Returns result as string."""
    try:
        expr = parse_expr(
            expression,
            transformations=standard_transformations
            + (implicit_multiplication_application,),
        )
        result = expr.evalf()
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {e}"


@tool
def differentiate(expression: str, variable: str = "x") -> str:
    """Compute the derivative of an expression. Returns derivative as string."""
    try:
        var = Symbol(variable)
        expr = parse_expr(
            expression,
            transformations=standard_transformations
            + (implicit_multiplication_application,),
        )
        derivative = diff(expr, var)
        return f"Derivative: {derivative}"
    except Exception as e:
        return f"Error differentiating: {e}"


@tool
def integrate_expr(expression: str, variable: str = "x") -> str:
    """Compute the integral of an expression. Returns integral as string."""
    try:
        var = Symbol(variable)
        expr = parse_expr(
            expression,
            transformations=standard_transformations
            + (implicit_multiplication_application,),
        )
        integral = integrate(expr, var)
        return f"Integral: {integral}"
    except Exception as e:
        return f"Error integrating: {e}"


@tool
def simplify_expr(expression: str) -> str:
    """Simplify a mathematical expression. Returns simplified expression as string."""
    try:
        expr = parse_expr(
            expression,
            transformations=standard_transformations
            + (implicit_multiplication_application,),
        )
        simplified = simplify(expr)
        return f"Simplified: {simplified}"
    except Exception as e:
        return f"Error simplifying: {e}"


def create_solver_agent(
    context_docs: list = None,
    session_id: str = None,
    user_id: str = None,
    model_obj: Any = None,
) -> Agent:
    """Create and return the Solver Agent. model_obj: optional model object override."""
    if model_obj is None:
        from config.models import get_agent_model

        model_obj = get_agent_model()

    context = ""
    if context_docs:
        context = "\n\n**Retrieved Knowledge:**\n" + "\n".join(
            [f"- {doc[:200]}..." for doc in context_docs[:3]]
        )

    instructions = f"""You are a Math Solver Agent. Solve math problems step-by-step.

Use the provided tools when needed:
- solve_equation: For algebraic equations
- differentiate: For calculus derivatives
- integrate_expr: For calculus integrals
- calculate: For numerical evaluation
- simplify_expr: For simplifying expressions

{context}

Provide a clear, step-by-step solution. Show your work and explain each step.
Always show which tools you're using and their outputs.
"""
    db = _get_solver_db()
    return Agent(
        name="Solver Agent",
        model=model_obj,
        tools=[solve_equation, calculate, differentiate, integrate_expr, simplify_expr],
        instructions=instructions,
        markdown=True,
        db=db,  # Required for add_history_to_context
        add_history_to_context=True,
        num_history_runs=3,
        session_id=session_id or "default",
        debug_mode=False,
    )


def solve_problem(
    parsed: dict,
    docs: list,
    memory_context: list = None,
    session_id: str = "",
    user_id: str = "",
    selected_model: str = None,
) -> tuple:
    """
    Solve the math problem using an agent with tool calling.
    selected_model: optional user-selected model ID (provider:model format).
    Returns: (solution, tool_calls)
    """
    from agno.agent import Agent
    import os

    problem_text = parsed.get("problem_text", "")

    # Separate RAG docs from web search results
    rag_docs = [d for d in docs if not str(d).startswith("Web Search Results:")]
    web_docs = [d for d in docs if str(d).startswith("Web Search Results:")]

    # Build comprehensive prompt with clear instructions
    prompt = f"""Solve this math problem step by step:

Problem: {problem_text}
"""

    if parsed.get("variables"):
        prompt += f"Variables: {', '.join(parsed.get('variables', []))}\n"

    if parsed.get("constraints"):
        prompt += f"Constraints: {', '.join(parsed.get('constraints', []))}\n"

    # Include RAG knowledge base documents
    if rag_docs:
        prompt += "\n**Relevant Knowledge from Knowledge Base:**\n"
        for i, doc in enumerate(rag_docs[:5], 1):
            doc_str = str(doc)
            prompt += f"{i}. {doc_str[:500]}{'...' if len(doc_str) > 500 else ''}\n"

    # Include web search results separately
    if web_docs:
        prompt += "\n**Additional Information from Web Search:**\n"
        for i, web_doc in enumerate(web_docs, 1):
            web_content = str(web_doc).replace("Web Search Results:\n", "").strip()
            prompt += (
                f"{i}. {web_content[:800]}{'...' if len(web_content) > 800 else ''}\n"
            )

    if memory_context:
        prompt += "\n**Similar Past Problems (for reference):**\n"
        for sim, past_q, past_sol in memory_context:
            prompt += f"\n- Question: {past_q}\n  Solution: {past_sol[:200]}...\n"

    prompt += """
**Instructions:**
- Analyze the problem carefully and use the provided knowledge to guide your solution
- Show your work step by step with clear mathematical reasoning
- Explain each step and why you're taking it
- Use proper mathematical notation and formatting
- Provide a complete solution with final answer
"""

    TOOLS = [solve_equation, calculate, differentiate, integrate_expr, simplify_expr]

    from config.models import get_agent_model
    from agents.web_search import search_web, search_wikipedia
    from agno.tools import tool

    model_obj = get_agent_model(selected=selected_model)

    # Add web search as a tool for the solver
    @tool
    def search_web_tool(query: str) -> str:
        """Search the web for additional mathematical information or examples. Use when you need more context or examples."""
        try:
            result = search_web(query, topic=parsed.get("topic"), max_results=3)
            return result if result else "No web results found."
        except Exception as e:
            return f"Web search error: {e}"

    # Add Wikipedia search as a tool for the solver
    @tool
    def search_wikipedia_tool(query: str) -> str:
        """Search Wikipedia for mathematical concepts, definitions, theorems, or formulas. Use when you need authoritative definitions or explanations."""
        try:
            result = search_wikipedia(query)
            return result if result else "No Wikipedia results found."
        except Exception as e:
            return f"Wikipedia search error: {e}"

    ALL_TOOLS = TOOLS + [search_web_tool, search_wikipedia_tool]

    # Always try with tools first - they're essential for math problems
    try:
        agent = Agent(
            name="Math Solver",
            model=model_obj,
            tools=ALL_TOOLS,
            markdown=True,
            instructions="""You are a math solver. You MUST use the provided tools when performing mathematical operations.

**CRITICAL: Use structured tool calls only. DO NOT write function calls as text like <function=...> or [function(...)]. The system will automatically call tools when you use them properly.**

**Mathematical Tools:**
- differentiate(expression, variable): Compute derivatives
- integrate_expr(expression, variable): Compute integrals  
- solve_equation(equation, variable): Solve algebraic equations
- calculate(expression): Evaluate numerical expressions
- simplify_expr(expression): Simplify mathematical expressions

**Search Tools:**
- search_web_tool(query): Search the web for additional context, examples, or formulas when needed
- search_wikipedia_tool(query): Search Wikipedia for mathematical concepts, definitions, theorems, or formulas

**Instructions:**
- ALWAYS use tools for mathematical operations (don't try to calculate manually)
- Use differentiate/integrate_expr for calculus problems
- Use solve_equation for algebraic problems
- Use calculate for numerical evaluation
- Use simplify_expr to simplify complex expressions
- Use search_web_tool if you need additional context or examples from the web
- Use search_wikipedia_tool if you need authoritative definitions, theorems, or mathematical concepts
- Show your work step by step, calling tools and explaining results
- Provide a complete solution with final answer
- Write your solution naturally - the tools will be called automatically when you reference them""",
        )

        response = agent.run(prompt)
        print("✅ Solved with tools")

    except Exception as error:
        error_msg = str(error)

        # Check for tool calling errors specifically
        if "tool_use_failed" in error_msg or "Failed to call a function" in error_msg:
            print(f"⚠️ Tool calling failed, retrying with math tools only...")
            # Retry with only math tools (no web search tool)
            try:
                agent_math_only = Agent(
                    name="Math Solver",
                    model=model_obj,
                    tools=TOOLS,  # Only math tools, no web search
                    markdown=True,
                    instructions="""You are a math solver. Use the provided mathematical tools when needed.

**CRITICAL: Use structured tool calls only. DO NOT write function calls as text. The system will automatically call tools when you use them properly.**

Available tools:
- differentiate(expression, variable): For derivatives
- integrate_expr(expression, variable): For integrals
- solve_equation(equation, variable): For solving equations
- calculate(expression): For numerical evaluation
- simplify_expr(expression): For simplification

Show your work step by step. Write your solution naturally - tools will be called automatically when needed.""",
                )
                response = agent_math_only.run(prompt)
                print("✅ Solved with math tools only (fallback mode)")
            except Exception as fallback_error:
                error_msg_fallback = str(fallback_error)
                # If tool calling still fails, try without tools as last resort
                if "tool_use_failed" in error_msg_fallback or "Failed to call a function" in error_msg_fallback:
                    print(f"⚠️ Tool calling still failing, trying without tools...")
                    try:
                        agent_no_tools = Agent(
                            name="Math Solver",
                            model=model_obj,
                            tools=[],  # No tools - solve directly
                            markdown=True,
                            instructions="""You are a math solver. Solve the problem step by step using mathematical reasoning.

Since tool calling is not available, solve the problem directly using your mathematical knowledge.
Show all steps clearly and provide the final answer.""",
                        )
                        response = agent_no_tools.run(prompt)
                        print("✅ Solved without tools (last resort mode)")
                    except Exception as no_tools_error:
                        error_msg = str(no_tools_error)
                        if (
                            "rate_limit" in error_msg.lower()
                            or "429" in error_msg
                            or "quota" in error_msg.lower()
                        ):
                            return {
                                "error": {
                                    "message": "Rate limit exceeded. Please try again in a few moments.",
                                    "type": "rate_limit",
                                }
                            }, []
                        else:
                            print(f"❌ Solver error (all modes failed): {no_tools_error}")
                            return {
                                "error": {
                                    "message": f"Unable to solve: {str(no_tools_error)}. Please try a different model or rephrase your question.",
                                    "type": "solver_error",
                                }
                            }, []
                elif (
                    "rate_limit" in error_msg_fallback.lower()
                    or "429" in error_msg_fallback
                    or "quota" in error_msg_fallback.lower()
                ):
                    return {
                        "error": {
                            "message": "Rate limit exceeded. Please try again in a few moments.",
                            "type": "rate_limit",
                        }
                    }, []
                else:
                    print(f"❌ Solver error (with fallback): {fallback_error}")
                    return {
                        "error": {
                            "message": str(fallback_error),
                            "type": "solver_error",
                        }
                    }, []
        elif (
            "rate_limit" in error_msg.lower()
            or "429" in error_msg
            or "quota" in error_msg.lower()
        ):
            return {
                "error": {
                    "message": "Rate limit exceeded. Please try again in a few moments.",
                    "type": "rate_limit",
                }
            }, []
        else:
            print(f"❌ Solver error: {error}")
            return {"error": {"message": str(error), "type": "solver_error"}}, []

    tool_calls = []
    solution = response.content

    # Extract tool calls from the response
    if hasattr(response, "messages") and response.messages:
        for msg in response.messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    info = {"tool": "unknown", "arguments": {}}
                    if hasattr(tc, "function"):
                        if hasattr(tc.function, "name"):
                            info["tool"] = tc.function.name
                        if hasattr(tc.function, "arguments"):
                            info["arguments"] = tc.function.arguments
                    elif isinstance(tc, dict):
                        fn = tc.get("function", {})
                        info["tool"] = fn.get("name", "unknown")
                        info["arguments"] = fn.get("arguments", {})
                    tool_calls.append(info)

    return solution, tool_calls
