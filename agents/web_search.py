"""
Web Search: DuckDuckGo fallback when RAG is insufficient.
"""

from agno.tools.duckduckgo import DuckDuckGoTools


def search_web(query: str, topic: str = None, max_results: int = 3) -> str:
    """Search the web for math-related information. Returns formatted results."""
    try:
        ddg = DuckDuckGoTools()
        search_query = f"math {topic} {query}" if topic else f"math {query}"
        results = ddg.duckduckgo_search(search_query, max_results=max_results)

        if not results:
            return "No web results found."

        if isinstance(results, str):
            return results

        formatted = []
        for i, result in enumerate(results[:max_results], 1):
            if isinstance(result, dict):
                title = result.get("title", "Unknown")
                url = result.get("url", "")
                snippet = result.get("snippet", result.get("body", ""))
                formatted.append(
                    f"**[{i}] {title}**\n**URL**: {url}\n**Content**: {snippet}"
                )
            else:
                formatted.append(f"**[{i}]** {result}")

        return "\n---\n".join(formatted)
    except Exception as e:
        return f"Web search error: {e}"


def search_wikipedia(query: str) -> str:
    """Search Wikipedia for math concepts."""
    try:
        from agno.tools.wikipedia import WikipediaTools

        wiki = WikipediaTools()
        return wiki.search_wikipedia(query)
    except Exception as e:
        return f"Wikipedia search error: {e}"
