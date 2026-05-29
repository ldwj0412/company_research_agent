from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from agents.structured_output import apply_tool_error_context, parse_specialist_output
from observability import tool_events_from_react_result
from state import AgentState
from tools.yfinance_tools import get_financial_summary, get_income_statement, get_price_history_summary

_SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "fundamental.txt").read_text(encoding="utf-8")

_primary = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)
_fallback = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)
_llm = _primary.with_fallbacks([_fallback])

_agent = create_react_agent(
    _llm,
    tools=[get_financial_summary, get_income_statement, get_price_history_summary],
    prompt=_SYSTEM_PROMPT,
)


def fundamental_node(state: AgentState) -> dict:
    ticker = state["ticker"]
    planner = state.get("planner") or {}
    rewritten_task = planner.get("rewritten_task") or f"Analyse the financial health of {ticker}."
    result = _agent.invoke({
        "messages": [("user", f"{rewritten_task}\nTicker: {ticker}. Use all available financial tools.")]
    })
    events = tool_events_from_react_result("fundamental", result)
    output = parse_specialist_output(result["messages"][-1].content)
    return {
        "fundamental_data": apply_tool_error_context(output, events),
        "run_events": events,
    }
