from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from state import AgentState
from tools.tavily_tools import get_tavily_tool

_SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "news_sentiment.txt").read_text(encoding="utf-8")

_primary = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)
_fallback = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)
_llm = _primary.with_fallbacks([_fallback])

_agent = create_react_agent(
    _llm,
    tools=[get_tavily_tool(max_results=8)],
    prompt=_SYSTEM_PROMPT,
)


def news_sentiment_node(state: AgentState) -> dict:
    ticker = state["ticker"]
    company_input = state.get("company_input", ticker)
    result = _agent.invoke({
        "messages": [("user", (
            f"Find recent news, management changes, regulatory risks, and analyst sentiment for "
            f"{company_input} (ticker: {ticker}). Cover the last 6 months."
        ))]
    })
    return {"news_data": result["messages"][-1].content}
