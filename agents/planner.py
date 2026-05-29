import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.structured_output import normalize_content
from state import AgentState, PlannerOutput


_ALLOWED_AGENTS = {"fundamental", "business_model", "news_sentiment"}
_planner_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)


def planner_node(state: AgentState) -> dict:
    user_input = state["company_input"]
    session_memory = state.get("session_memory") or {}
    guardrail_plan = _guardrail_plan(user_input, session_memory)
    if guardrail_plan:
        return {"planner": guardrail_plan}

    response = _planner_llm.invoke([
        SystemMessage(content=(
            "You are a constrained planning router for a company research agent. "
            "Return JSON only. Choose which specialist agents are needed. "
            "Do not invent tool names. Allowed agents: fundamental, business_model, news_sentiment. "
            "Use the fewest specialist agents that can answer the query well; do not run all agents by default. "
            "Use answer_style five_section_report only for a full company brief. "
            "Use short_answer for follow-ups or narrow questions. "
            "Agent selection defaults: financial_only uses [\"fundamental\"] and short_answer; "
            "business_model_only uses [\"business_model\"] and short_answer; "
            "news_only or recent_developments uses [\"news_sentiment\"] and short_answer; "
            "risk_followup usually uses [\"fundamental\", \"news_sentiment\"] and short_answer, "
            "adding business_model only when the user asks about competitive position, concentration, or moat. "
            "Only full_report should normally use all three agents and five_section_report. "
            "If the user omits a company but session memory has last_ticker, set reuse_last_ticker true, "
            "leave target_company empty, and answer as a follow-up about the remembered company."
        )),
        HumanMessage(content=(
            f"User query: {user_input}\n"
            f"Session memory: {json.dumps(session_memory, ensure_ascii=False)}\n\n"
            "Return this JSON shape exactly:\n"
            "{\n"
            '  "intent": "full_report | risk_followup | business_model_only | financial_only | news_only | recent_developments | general_followup",\n'
            '  "target_company": "company name or empty string",\n'
            '  "reuse_last_ticker": false,\n'
            '  "required_agents": ["fundamental", "business_model", "news_sentiment"],\n'
            '  "answer_style": "five_section_report",\n'
            '  "rewritten_task": "clear instruction for downstream specialists and writer"\n'
            "}"
        )),
    ])

    return {"planner": parse_planner_output(response.content, user_input, session_memory)}


def _guardrail_plan(user_input: str, session_memory: dict) -> PlannerOutput | None:
    if _looks_like_bare_company_query(user_input):
        return {
            "intent": "full_report",
            "target_company": user_input.strip(),
            "reuse_last_ticker": False,
            "required_agents": ["fundamental", "business_model", "news_sentiment"],
            "answer_style": "five_section_report",
            "rewritten_task": f"Create a full company research report for: {user_input.strip()}",
        }

    return None


def _looks_like_bare_company_query(user_input: str) -> bool:
    cleaned = user_input.strip().strip("?.! ")
    if not cleaned:
        return False
    lowered = cleaned.lower()
    intent_terms = (
        "what",
        "who",
        "why",
        "how",
        "when",
        "where",
        "own",
        "owner",
        "owners",
        "supplier",
        "suppliers",
        "risk",
        "risks",
        "news",
        "recent",
        "latest",
        "happened",
        "changed",
        "valuation",
        "expensive",
        "cheap",
        "make money",
        "business model",
        "financial",
        "healthy",
    )
    if any(term in lowered for term in intent_terms):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 .&'-]{0,40}", cleaned))


def parse_planner_output(content, user_input: str, session_memory: dict | None = None) -> PlannerOutput:
    raw_text = normalize_content(content)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return _fallback_plan(user_input, session_memory or {})

    if not isinstance(parsed, dict):
        return _fallback_plan(user_input, session_memory or {})

    answer_style = str(parsed.get("answer_style") or "").strip()
    if answer_style not in {"five_section_report", "short_answer"}:
        answer_style = "five_section_report"

    return {
        "intent": str(parsed.get("intent") or "full_report").strip() or "full_report",
        "target_company": str(parsed.get("target_company") or "").strip(),
        "reuse_last_ticker": bool(parsed.get("reuse_last_ticker")),
        "required_agents": _valid_agents(parsed.get("required_agents")),
        "answer_style": answer_style,
        "rewritten_task": str(parsed.get("rewritten_task") or user_input).strip() or user_input,
    }


def _fallback_plan(user_input: str, session_memory: dict) -> PlannerOutput:
    lowered = user_input.lower()
    has_last_ticker = bool(session_memory.get("last_ticker"))
    follow_up_terms = ("risk", "risks", "why", "what about", "how about", "recent", "news", "valuation")

    if has_last_ticker and any(term in lowered for term in follow_up_terms):
        return {
            "intent": "general_followup",
            "target_company": "",
            "reuse_last_ticker": True,
            "required_agents": [],
            "answer_style": "short_answer",
            "rewritten_task": user_input,
        }

    return {
        "intent": "full_report",
        "target_company": user_input,
        "reuse_last_ticker": False,
        "required_agents": ["fundamental", "business_model", "news_sentiment"],
        "answer_style": "five_section_report",
        "rewritten_task": f"Create a full company research report for: {user_input}",
    }


def _valid_agents(value) -> list[str]:
    if not isinstance(value, list):
        return ["fundamental", "business_model", "news_sentiment"]
    agents = [str(item).strip() for item in value if str(item).strip() in _ALLOWED_AGENTS]
    return agents
