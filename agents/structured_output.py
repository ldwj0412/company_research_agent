import json
from typing import Any, Optional

from state import SpecialistOutput

_VALID_DATA_QUALITY = {"good", "partial", "weak"}
_VALID_CONFIDENCE = {"high", "medium", "low"}


def normalize_content(content: Any) -> str:
    """Normalize LLM content that may arrive as a string or content-part list."""
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    return str(content or "").strip()


def parse_specialist_output(content: Any) -> SpecialistOutput:
    """Parse a specialist's strict JSON response, falling back safely on errors."""
    raw_text = normalize_content(content)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return _fallback_output(raw_text)

    if not isinstance(parsed, dict):
        return _fallback_output(raw_text)

    summary = str(parsed.get("summary") or _fallback_summary(raw_text)).strip()
    return {
        "summary": summary,
        "key_facts": _string_list(parsed.get("key_facts")),
        "risks": _string_list(parsed.get("risks")),
        "sources": _string_list(parsed.get("sources")),
        "data_quality": _validated_choice(parsed.get("data_quality"), _VALID_DATA_QUALITY, "weak"),
        "confidence": _validated_choice(parsed.get("confidence"), _VALID_CONFIDENCE, "low"),
        "raw_text": raw_text,
    }


def format_specialist_context(label: str, output: Optional[SpecialistOutput]) -> str:
    """Format structured specialist output for the final report writer."""
    if output is None:
        output = _fallback_output("")

    return f"""--- {label.upper()} DATA ---
Data quality: {output["data_quality"]}
Confidence: {output["confidence"]}

Summary: {output["summary"]}

Key facts:
{_format_list(output["key_facts"])}

Risks/Caveats:
{_format_list(output["risks"])}

Sources:
{_format_list(output["sources"])}

Raw fallback/context:
{output["raw_text"] or "N/A"}"""


def _fallback_output(raw_text: str) -> SpecialistOutput:
    return {
        "summary": _fallback_summary(raw_text),
        "key_facts": [],
        "risks": [],
        "sources": [],
        "data_quality": "weak",
        "confidence": "low",
        "raw_text": raw_text,
    }


def _fallback_summary(raw_text: str) -> str:
    if not raw_text:
        return "Structured output unavailable."
    first_line = next((line.strip() for line in raw_text.splitlines() if line.strip()), raw_text)
    return first_line[:300]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _validated_choice(value: Any, valid_values: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in valid_values else default


def _format_list(items: list[str]) -> str:
    if not items:
        return "- None provided."
    return "\n".join(f"- {item}" for item in items)
