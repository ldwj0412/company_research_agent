import json

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

from tools.retry import retry_transient


def get_tavily_tool(max_results: int = 5, sleep_seconds: float = 0.5):
    search_tool = TavilySearchResults(max_results=max_results)

    @tool("tavily_search_results_json")
    def tavily_search_results_json(query: str) -> str:
        """Search the web for recent company information and return JSON results."""
        result = retry_transient(
            lambda: search_tool.invoke({"query": query}),
            max_attempts=2,
            sleep_seconds=sleep_seconds,
        )
        if result.error:
            return result.format_failure("Tavily search")
        return _format_tool_value(result.value, result.attempts)

    return tavily_search_results_json


def _format_tool_value(value, attempts: int) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    if attempts > 1:
        return f"Tool retry note: succeeded after {attempts} attempts.\n{text}"
    return text
