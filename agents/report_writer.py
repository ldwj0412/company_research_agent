from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.structured_output import format_specialist_context, normalize_content
from observability import format_run_events, make_run_event
from state import AgentState, SpecialistOutput

_SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "report_writer.txt").read_text(encoding="utf-8")

_PRIMARY_MODEL = "gemini-3.1-flash-lite"
_FALLBACK_MODEL = "gemini-2.5-flash-lite"
_primary = ChatGoogleGenerativeAI(model=_PRIMARY_MODEL, temperature=0.3)
_fallback = ChatGoogleGenerativeAI(model=_FALLBACK_MODEL, temperature=0.3)


def report_writer_node(state: AgentState) -> dict:
    ticker = state.get("ticker", "")
    company_input = state.get("company_input", ticker)
    planner = state.get("planner") or {}
    answer_style = planner.get("answer_style", "five_section_report")

    combined = f"""COMPANY: {company_input} (ticker: {ticker})
USER QUESTION: {company_input}
ANSWER STYLE: {answer_style}
PLANNER INTENT: {planner.get("intent", "full_report")}
REWRITTEN TASK: {planner.get("rewritten_task", company_input)}

--- PREVIOUS SESSION REPORT ---
{state.get("previous_report") or "None available."}

{_format_routed_specialist_context(state)}

--- REQUIRED UNCERTAINTY NOTES ---
{_format_uncertainty_notes(state)}

--- RUN OBSERVABILITY ---
{format_run_events(state.get("run_events"))}

Write the research report now."""

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=combined),
    ]
    content, event = _invoke_with_fallback(messages)
    return {"report": content, "run_events": [event]}


def _invoke_with_fallback(messages: list[SystemMessage | HumanMessage]) -> tuple[str, dict]:
    import time

    started = time.perf_counter()
    primary_error = None
    try:
        response = _primary.invoke(messages)
        return normalize_content(response.content), make_run_event(
            node="report_writer",
            status="success",
            started=started,
            model=_PRIMARY_MODEL,
            fallback_model=_FALLBACK_MODEL,
        )
    except Exception as exc:
        primary_error = exc

    try:
        response = _fallback.invoke(messages)
        return normalize_content(response.content), make_run_event(
            node="report_writer",
            status="success",
            started=started,
            error=f"Primary model failed: {primary_error}",
            model=_FALLBACK_MODEL,
            fallback_model=_FALLBACK_MODEL,
            fallback_used=True,
        )
    except Exception as fallback_error:
        event = make_run_event(
            node="report_writer",
            status="error",
            started=started,
            error=f"Primary model failed: {primary_error}; fallback model failed: {fallback_error}",
            model=_FALLBACK_MODEL,
            fallback_model=_FALLBACK_MODEL,
            fallback_used=True,
        )
        return _emergency_report(), event


def _emergency_report() -> str:
    return """## 1. What Does This Company Do?
Unable to generate a reliable company description because the report-writing model was unavailable.

## 2. How Does It Make Money?
Unable to generate a reliable revenue-model summary from the available context.

## 3. Is the Business Healthy?
Unable to synthesize the financial-health section. Please retry once the model service is available.

## 4. What Are the Risks?
* The report-writing model failed, so this run should not be used as investment research.
* Retry the request after checking model/API configuration.

## 5. One-Line Verdict
No investment verdict is available for this run.

Data note: Report generation failed after both the primary and fallback models were unavailable."""


def _format_uncertainty_notes(state: AgentState) -> str:
    notes = []
    planner = state.get("planner") or {}
    required_agents = planner.get("required_agents")
    include_missing = planner.get("answer_style", "five_section_report") == "five_section_report"
    if required_agents is None:
        required_agents = ["fundamental", "business_model", "news_sentiment"]

    for agent_name, label, output in [
        ("fundamental", "Fundamental", state.get("fundamental_data")),
        ("business_model", "Business model", state.get("business_model_data")),
        ("news_sentiment", "News & sentiment", state.get("news_data")),
    ]:
        if agent_name not in required_agents and not include_missing:
            continue
        note = _uncertainty_note(label, output)
        if note:
            notes.append(note)

    if not notes:
        return "- None. Do not add a Data note."

    if planner.get("answer_style", "five_section_report") == "short_answer":
        return (
            "Group these notes into one concise Data note at the end of the answer. "
            "Do not expose raw tool errors.\n"
            + "\n".join(notes)
        )

    return (
        "Group these notes into one concise Data note at the end of section 5, after the verdict sentence. "
        "Do not repeat these caveats throughout sections 1-4. Do not add a sixth section or expose raw tool errors.\n"
        + "\n".join(notes)
    )


def _format_routed_specialist_context(state: AgentState) -> str:
    planner = state.get("planner") or {}
    answer_style = planner.get("answer_style", "five_section_report")
    required_agents = planner.get("required_agents")

    sections = []
    for agent_name, label, key in [
        ("fundamental", "Fundamental", "fundamental_data"),
        ("business_model", "Business model", "business_model_data"),
        ("news_sentiment", "News & sentiment", "news_data"),
    ]:
        if answer_style == "short_answer" and required_agents is not None and agent_name not in required_agents:
            continue
        sections.append(format_specialist_context(label, state.get(key)))

    if not sections:
        return "--- SPECIALIST DATA ---\nNo new specialist data was requested for this follow-up."
    return "\n\n".join(sections)


def _uncertainty_note(
    label: str,
    output: SpecialistOutput | None,
) -> str:
    if output is None:
        return f"- {label}: No structured context was available."

    weak_quality = output["data_quality"] == "weak"
    low_confidence = output["confidence"] == "low"
    no_traceable_sources = not any(str(source).startswith(("http://", "https://")) for source in output["sources"])

    if not (weak_quality or low_confidence):
        return ""

    reason = "evidence is limited"
    if weak_quality and low_confidence:
        reason = "evidence quality is weak and confidence is low"
    elif weak_quality:
        reason = "evidence quality is weak"
    elif low_confidence:
        reason = "confidence is low"

    source_note = " No traceable source URLs were available." if no_traceable_sources else ""
    return (
        f"- {label}: {reason}.{source_note} "
        "Use plain-English wording for readers."
    )
