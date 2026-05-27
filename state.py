from operator import add
from typing import Annotated, Literal, Optional, TypedDict


DataQuality = Literal["good", "partial", "weak"]
Confidence = Literal["high", "medium", "low"]


class SpecialistOutput(TypedDict):
    summary: str
    key_facts: list[str]
    risks: list[str]
    sources: list[str]
    data_quality: DataQuality
    confidence: Confidence
    raw_text: str


class RunEvent(TypedDict):
    node: str
    status: str
    latency_ms: int
    error: str
    error_category: str
    model: str
    fallback_model: str
    fallback_used: bool


class AgentState(TypedDict):
    company_input: str           # raw user input e.g. "Apple" or "help me research Tencent"
    ticker: str                  # resolved ticker e.g. "AAPL", "0700.HK"
    fundamental_data: Optional[SpecialistOutput]
    business_model_data: Optional[SpecialistOutput]
    news_data: Optional[SpecialistOutput]
    report: Optional[str]
    report_cache: Optional[dict[str, str]]
    cache_hit: Optional[bool]
    error: Optional[str]
    run_events: Annotated[list[RunEvent], add]
