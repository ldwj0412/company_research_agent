from urllib.parse import urlparse


_PLACEHOLDER_SOURCES = {"", "n/a", "none", "not available", "unknown"}
_HIGH_TRUST_DOMAINS = (
    "sec.gov",
    "www.sec.gov",
    "investor.apple.com",
    "abc.xyz",
    "investor.nvidia.com",
    "microsoft.com",
    "ir.tencent.com",
)


def rank_sources(sources: list[str]) -> list[dict[str, str | int]]:
    """Deduplicate and rank sources so report synthesis can prefer traceable evidence."""
    seen: set[str] = set()
    ranked = []

    for source in sources:
        cleaned = str(source or "").strip()
        if cleaned.lower() in _PLACEHOLDER_SOURCES:
            continue

        key = cleaned.lower().rstrip("/")
        if key in seen:
            continue
        seen.add(key)

        parsed = urlparse(cleaned)
        is_url = parsed.scheme in {"http", "https"} and bool(parsed.netloc)
        domain = parsed.netloc.lower()
        score = _score_source(is_url, domain)
        ranked.append({
            "source": cleaned,
            "traceability": "url" if is_url else "named_source",
            "domain": domain,
            "score": score,
        })

    return sorted(ranked, key=lambda item: int(item["score"]), reverse=True)


def format_source_quality(sources: list[str]) -> str:
    ranked = rank_sources(sources)
    traceable = [item for item in ranked if item["traceability"] == "url"]

    if not ranked:
        return "No usable sources were provided. Treat search-based claims as weak unless supported elsewhere."

    if traceable:
        return (
            f"{len(ranked)} usable source(s), including {len(traceable)} traceable URL(s). "
            "Use source-backed claims first and treat unsourced claims cautiously."
        )

    return (
        f"{len(ranked)} named source(s), but no traceable URLs. "
        "Use this context cautiously and avoid unsupported precise claims."
    )


def format_traceable_sources(sources: list[str]) -> str:
    traceable_sources = [
        str(item["source"])
        for item in rank_sources(sources)
        if item["traceability"] == "url"
    ]
    if not traceable_sources:
        return "- None provided."
    return "\n".join(f"- {source}" for source in traceable_sources)


def _score_source(is_url: bool, domain: str) -> int:
    if not is_url:
        return 10
    if domain in _HIGH_TRUST_DOMAINS or domain.endswith(".gov"):
        return 100
    if "investor" in domain or domain.endswith(".com"):
        return 80
    return 60
