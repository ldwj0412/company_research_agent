from typing import Optional, TypedDict


class AgentState(TypedDict):
    company_input: str           # raw user input e.g. "Apple" or "help me research Tencent"
    ticker: str                  # resolved ticker e.g. "AAPL", "0700.HK"
    fundamental_data: Optional[str]
    business_model_data: Optional[str]
    news_data: Optional[str]
    report: Optional[str]
    report_cache: Optional[dict[str, str]]
    cache_hit: Optional[bool]
    error: Optional[str]
