from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from agents.structured_output import apply_tool_error_context, parse_specialist_output
from observability import tool_events_from_react_result
from state import AgentState
from tools.tavily_tools import get_tavily_tool

_SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "business_model.txt").read_text(encoding="utf-8")

_primary = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)
_fallback = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)
_llm = _primary.with_fallbacks([_fallback])

_agent = create_react_agent(
    _llm,
    tools=[get_tavily_tool(max_results=5)],
    prompt=_SYSTEM_PROMPT,
)


def business_model_node(state: AgentState) -> dict:
    ticker = state["ticker"]
    company_input = state.get("company_input", ticker)
    result = _agent.invoke({
        "messages": [("user", (
            f"Research the business model and competitive position of {company_input} (ticker: {ticker}). "
            "Search for how it makes money, its main revenue streams, competitive advantages, and geographic exposure."
        ))]
    })
    events = tool_events_from_react_result("business_model", result)
    output = parse_specialist_output(result["messages"][-1].content)
    return {
        "business_model_data": apply_tool_error_context(output, events),
        "run_events": events,
    }
