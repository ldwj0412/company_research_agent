from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

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
    result = _agent.invoke({
        "messages": [("user", f"Analyse the financial health of {ticker}. Use all available tools.")]
    })
    return {"fundamental_data": result["messages"][-1].content}
