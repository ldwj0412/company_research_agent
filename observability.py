import json
import os
import sys
import time
from collections.abc import Callable
from typing import Any


def trace_node(
    node_name: str,
    func: Callable[[dict[str, Any]], dict[str, Any]],
    fallback_output: dict[str, Any] | Callable[[Exception, dict[str, Any]], dict[str, Any]] | None = None,
    model: str = "",
    fallback_model: str = "",
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Wrap a graph node and append a structured run event."""
    def wrapped(state: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            result = func(state)
            event = make_run_event(
                node=node_name,
                status="success",
                started=started,
                model=model,
                fallback_model=fallback_model,
            )
        except Exception as exc:
            result = _resolve_fallback(fallback_output, exc, state)
            event = make_run_event(
                node=node_name,
                status="error",
                started=started,
                error=str(exc),
                model=model,
                fallback_model=fallback_model,
            )

        _emit_run_event(event)
        return _append_run_event(result, event)

    return wrapped


def make_run_event(
    node: str,
    status: str,
    started: float,
    error: str = "",
    model: str = "",
    fallback_model: str = "",
    fallback_used: bool = False,
) -> dict[str, Any]:
    return {
        "node": node,
        "status": status,
        "latency_ms": max(0, int((time.perf_counter() - started) * 1000)),
        "error": error,
        "error_category": classify_error(error) if status == "error" else "",
        "model": model,
        "fallback_model": fallback_model,
        "fallback_used": fallback_used,
    }


def format_run_events(events: list[dict[str, Any]] | None) -> str:
    if not events:
        return "- No run events recorded."

    lines = []
    for event in events:
        line = (
            f"- {event.get('node', 'unknown')}: {event.get('status', 'unknown')} "
            f"in {event.get('latency_ms', 'N/A')} ms"
        )
        if event.get("fallback_used"):
            line += f" (fallback model used: {event.get('model', 'unknown')})"
        elif event.get("fallback_model"):
            line += f" (fallback available: {event.get('fallback_model')})"
        if event.get("error"):
            line += f"; error: {event['error']}"
        lines.append(line)
    return "\n".join(lines)


def tool_events_from_react_result(node_name: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarize ReAct tool activity from returned messages when available."""
    messages = result.get("messages") or []
    requested = 0
    completed = 0
    tool_names: list[str] = []
    tool_errors: list[str] = []
    retry_notes: list[str] = []

    for message in messages:
        tool_calls = getattr(message, "tool_calls", None) or []
        requested += len(tool_calls)
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            if name:
                tool_names.append(str(name))

        message_type = getattr(message, "type", "")
        class_name = message.__class__.__name__.lower()
        if message_type == "tool" or class_name == "toolmessage":
            completed += 1
            name = getattr(message, "name", "")
            if name:
                tool_names.append(str(name))
            content = str(getattr(message, "content", "") or "")
            if _looks_like_tool_error(content):
                tool_errors.append(content[:300])
            retry_note = _extract_retry_note(content)
            if retry_note:
                retry_notes.append(retry_note)

    if requested == 0 and completed == 0:
        return []

    unique_tools = ", ".join(sorted(set(tool_names))) or "unknown"
    status = "error" if tool_errors else "success"
    error = f"Tool activity: {requested} requested, {completed} completed; tools: {unique_tools}"
    if retry_notes:
        error += f"; retries: {' | '.join(retry_notes)}"
    if tool_errors:
        error += f"; errors: {' | '.join(tool_errors)}"
    return [{
        "node": f"{node_name}.tools",
        "status": status,
        "latency_ms": 0,
        "error": error,
        "error_category": classify_error(error) if status == "error" else "",
        "model": "",
        "fallback_model": "",
        "fallback_used": False,
    }]


def has_error_event(events: list[dict[str, Any]]) -> bool:
    return any(event.get("status") == "error" for event in events)


def classify_error(error: str) -> str:
    text = str(error or "").lower()
    if not text:
        return ""
    if "401" in text or "unauthorized" in text or "api key" in text or "forbidden" in text:
        return "auth_error"
    if "429" in text or "rate limit" in text or "quota" in text:
        return "rate_limit"
    if "json" in text or "parse" in text or "malformed" in text:
        return "parse_error"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "model" in text or "gemini" in text or "llm" in text:
        return "model_error"
    if "http" in text or "client error" in text or "connection" in text:
        return "tool_error"
    return "unknown_error"


def _append_run_event(result: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    existing = result.get("run_events") or []
    return {**result, "run_events": existing + [event]}


def _resolve_fallback(
    fallback_output: dict[str, Any] | Callable[[Exception, dict[str, Any]], dict[str, Any]] | None,
    exc: Exception,
    state: dict[str, Any],
) -> dict[str, Any]:
    if callable(fallback_output):
        return fallback_output(exc, state)
    if fallback_output is not None:
        return dict(fallback_output)
    return {"error": str(exc)}


def _looks_like_tool_error(content: str) -> bool:
    markers = (
        "httperror",
        "client error",
        "unauthorized",
        "forbidden",
        "api key",
        "rate limit",
        "timeout",
        "connection error",
        "error fetching",
        "auth_error",
        "rate_limit",
        "tool_error",
    )
    lowered = content.lower()
    return any(marker in lowered for marker in markers)


def _extract_retry_note(content: str) -> str:
    first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
    if first_line.lower().startswith("tool retry note:"):
        return first_line.removeprefix("Tool retry note:").strip()
    return ""


def _emit_run_event(event: dict[str, Any]) -> None:
    if os.getenv("AGENT_TRACE", "").lower() not in {"1", "true", "yes"}:
        return
    print(json.dumps(event, ensure_ascii=True), file=sys.stderr)
