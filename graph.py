from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from agents.business_model import business_model_node
from agents.fundamental import fundamental_node
from agents.news_sentiment import news_sentiment_node
from agents.planner import planner_node
from agents.report_writer import report_writer_node
from agents.structured_output import parse_specialist_output
from observability import trace_node
from state import AgentState

_TICKER_MODEL = "gemini-3.1-flash-lite"
_FALLBACK_MODEL = "gemini-2.5-flash-lite"
_ticker_llm = ChatGoogleGenerativeAI(model=_TICKER_MODEL, temperature=0)


def orchestrate_node(state: AgentState) -> dict:
    """Resolve the user's natural-language input to a stock ticker symbol."""
    user_input = state["company_input"]
    planner = state.get("planner") or {}
    session_memory = state.get("session_memory") or {}
    report_cache = state.get("report_cache") or {}

    answer_style = planner.get("answer_style", "five_section_report")
    if planner.get("reuse_last_ticker") and session_memory.get("last_ticker"):
        ticker = str(session_memory["last_ticker"]).upper()
        previous_report = report_cache.get(ticker) or session_memory.get("last_report")
        return {
            "ticker": ticker,
            "previous_report": previous_report,
            "planner": planner,
            "cache_hit": False,
            "error": None,
        }

    query_for_ticker = planner.get("target_company") or user_input
    response = _ticker_llm.invoke([
        SystemMessage(content=(
            "You are a stock ticker resolver. "
            "The user may type a company name, a sentence, or a ticker. "
            "Reply with ONLY the stock ticker symbol (e.g. AAPL, 0700.HK, 9988.HK). "
            "For HK-listed stocks use the numeric code with .HK suffix. "
            "If you cannot identify the company, reply with UNKNOWN."
        )),
        HumanMessage(content=query_for_ticker),
    ])
    content = response.content
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    ticker = content.strip().upper()

    if ticker == "UNKNOWN" or not ticker:
        return {"ticker": "", "error": f"Could not identify a stock ticker for: {user_input}"}

    if ticker in report_cache and answer_style == "five_section_report":
        return {
            "ticker": ticker,
            "report": report_cache[ticker],
            "planner": planner,
            "cache_hit": True,
            "error": None,
        }

    return {
        "ticker": ticker,
        "previous_report": report_cache.get(ticker),
        "planner": planner,
        "cache_hit": False,
        "error": None,
    }


def route_to_agents(state: AgentState):
    """Return a list of agent node names to run in parallel, or END on error."""
    if state.get("error") or state.get("cache_hit") or state.get("report"):
        return END
    planner = state.get("planner") or {}
    required_agents = planner.get("required_agents")
    if required_agents is None:
        return ["fundamental", "business_model", "news_sentiment"]
    if not required_agents:
        return "report_writer"
    return required_agents


def build_graph():
    g = StateGraph(AgentState)

    g.add_node(
        "planner",
        trace_node(
            "planner",
            planner_node,
            fallback_output=lambda exc, state: {
                "planner": {
                    "intent": "full_report",
                    "target_company": state.get("company_input", ""),
                    "reuse_last_ticker": False,
                    "required_agents": ["fundamental", "business_model", "news_sentiment"],
                    "answer_style": "five_section_report",
                    "rewritten_task": state.get("company_input", ""),
                }
            },
            model=_TICKER_MODEL,
        ),
    )
    g.add_node(
        "orchestrate",
        trace_node(
            "orchestrate",
            orchestrate_node,
            fallback_output=lambda exc, state: {
                "ticker": "",
                "error": f"Could not resolve ticker: {exc}",
            },
            model=_TICKER_MODEL,
        ),
    )
    g.add_node(
        "fundamental",
        trace_node(
            "fundamental",
            fundamental_node,
            fallback_output=lambda exc, state: {
                "fundamental_data": parse_specialist_output(f"Fundamental agent failed: {exc}")
            },
            model=_TICKER_MODEL,
            fallback_model=_FALLBACK_MODEL,
        ),
    )
    g.add_node(
        "business_model",
        trace_node(
            "business_model",
            business_model_node,
            fallback_output=lambda exc, state: {
                "business_model_data": parse_specialist_output(f"Business model agent failed: {exc}")
            },
            model=_TICKER_MODEL,
            fallback_model=_FALLBACK_MODEL,
        ),
    )
    g.add_node(
        "news_sentiment",
        trace_node(
            "news_sentiment",
            news_sentiment_node,
            fallback_output=lambda exc, state: {
                "news_data": parse_specialist_output(f"News sentiment agent failed: {exc}")
            },
            model=_TICKER_MODEL,
            fallback_model=_FALLBACK_MODEL,
        ),
    )
    g.add_node("report_writer",  report_writer_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "orchestrate")

    # Fan-out: conditional edges returning a list triggers parallel execution
    g.add_conditional_edges(
        "orchestrate",
        route_to_agents,
        ["fundamental", "business_model", "news_sentiment", "report_writer", END],
    )

    # Fan-in: all 3 → report_writer (LangGraph waits for all 3 to finish)
    g.add_edge("fundamental",    "report_writer")
    g.add_edge("business_model", "report_writer")
    g.add_edge("news_sentiment", "report_writer")

    g.add_edge("report_writer", END)

    return g.compile()
