from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from state import AgentState

_SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "report_writer.txt").read_text(encoding="utf-8")

_primary = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.3)
_fallback = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)
_llm = _primary.with_fallbacks([_fallback])


def report_writer_node(state: AgentState) -> dict:
    ticker = state.get("ticker", "")
    company_input = state.get("company_input", ticker)

    combined = f"""COMPANY: {company_input} (ticker: {ticker})

--- FUNDAMENTAL DATA ---
{state.get('fundamental_data') or 'Not available.'}

--- BUSINESS MODEL DATA ---
{state.get('business_model_data') or 'Not available.'}

--- NEWS & SENTIMENT DATA ---
{state.get('news_data') or 'Not available.'}

Write the research report now."""

    response = _llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=combined),
    ])
    content = response.content
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return {"report": content}
