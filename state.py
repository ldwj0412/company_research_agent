from typing import Literal, Optional, TypedDict


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
