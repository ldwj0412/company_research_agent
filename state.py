from typing import TypedDict, Optional


class AgentState(TypedDict):
    company_input: str           # raw user input e.g. "Apple" or "help me research Tencent"
    ticker: str                  # resolved ticker e.g. "AAPL", "0700.HK"
    fundamental_data: Optional[str]
    business_model_data: Optional[str]
    news_data: Optional[str]
    report: Optional[str]
    error: Optional[str]
