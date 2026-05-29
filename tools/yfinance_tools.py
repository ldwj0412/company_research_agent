import yfinance as yf
from langchain_core.tools import tool

from tools.retry import retry_transient


_TOOL_RETRY_SLEEP_SECONDS = 0.5


def _fmt(val, fmt=".2f", suffix=""):
    if val is None:
        return "N/A"
    try:
        return f"{val:{fmt}}{suffix}"
    except Exception:
        return str(val)


def _millions(val):
    if val is None:
        return "N/A"
    try:
        return f"${val / 1_000_000:.0f}M"
    except Exception:
        return "N/A"


def _billions(val):
    if val is None:
        return "N/A"
    try:
        if val >= 1_000_000_000_000:
            return f"${val / 1_000_000_000_000:.2f}T"
        return f"${val / 1_000_000_000:.2f}B"
    except Exception:
        return "N/A"


@tool
def get_financial_summary(ticker: str) -> str:
    """Get key financial metrics for a stock ticker: valuation, profitability, balance sheet health."""
    try:
        result = retry_transient(
            lambda: yf.Ticker(ticker).info,
            max_attempts=2,
            sleep_seconds=_TOOL_RETRY_SLEEP_SECONDS,
        )
        if result.error:
            return result.format_failure(f"financial summary for {ticker}")
        info = result.value or {}
        name = info.get("longName") or info.get("shortName") or ticker
        lines = [
            f"Company: {name} ({ticker})",
            f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}",
            f"Market Cap: {_billions(info.get('marketCap'))}",
            "",
            "--- Valuation ---",
            f"P/E (trailing): {_fmt(info.get('trailingPE'))}",
            f"P/E (forward):  {_fmt(info.get('forwardPE'))}",
            f"Price/Book:     {_fmt(info.get('priceToBook'))}",
            f"EV/EBITDA:      {_fmt(info.get('enterpriseToEbitda'))}",
            "",
            "--- Profitability ---",
            f"Gross Margin:     {_fmt(info.get('grossMargins'), '.1%')}",
            f"Operating Margin: {_fmt(info.get('operatingMargins'), '.1%')}",
            f"Net Margin:       {_fmt(info.get('profitMargins'), '.1%')}",
            f"ROE:              {_fmt(info.get('returnOnEquity'), '.1%')}",
            f"ROA:              {_fmt(info.get('returnOnAssets'), '.1%')}",
            "",
            "--- Balance Sheet ---",
            f"Debt/Equity:   {_fmt(info.get('debtToEquity'))}",
            f"Current Ratio: {_fmt(info.get('currentRatio'))}",
            f"Total Cash:    {_billions(info.get('totalCash'))}",
            f"Total Debt:    {_billions(info.get('totalDebt'))}",
            f"Free Cash Flow:{_billions(info.get('freeCashflow'))}",
            "",
            "--- Growth ---",
            f"Revenue Growth (YoY): {_fmt(info.get('revenueGrowth'), '.1%')}",
            f"Earnings Growth:      {_fmt(info.get('earningsGrowth'), '.1%')}",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching financial summary for {ticker}: {e}"


@tool
def get_income_statement(ticker: str) -> str:
    """Get the last 2 years of annual income statement data: revenue, gross profit, operating income, net income."""
    try:
        result = retry_transient(
            lambda: yf.Ticker(ticker).financials,
            max_attempts=2,
            sleep_seconds=_TOOL_RETRY_SLEEP_SECONDS,
        )
        if result.error:
            return result.format_failure(f"income statement for {ticker}")
        fin = result.value
        if fin is None or fin.empty:
            return f"No income statement data available for {ticker}."

        rows = ["Annual Income Statement (last 2 years):"]
        metrics = {
            "Total Revenue": "Total Revenue",
            "Gross Profit": "Gross Profit",
            "Operating Income": "Operating Income",
            "Net Income": "Net Income",
        }
        cols = fin.columns[:2]  # most recent 2 years

        header = "Metric".ljust(22) + "  ".join(str(c.year) for c in cols)
        rows.append(header)
        rows.append("-" * 50)

        for display, key in metrics.items():
            if key in fin.index:
                vals = [_billions(fin.loc[key, c]) for c in cols]
                rows.append(f"{display.ljust(22)}{('  ').join(vals)}")
            else:
                rows.append(f"{display.ljust(22)}N/A")

        return "\n".join(rows)
    except Exception as e:
        return f"Error fetching income statement for {ticker}: {e}"


@tool
def get_price_history_summary(ticker: str) -> str:
    """Get 52-week price range, YTD performance, and average volume for a stock ticker."""
    try:
        result = retry_transient(
            lambda: yf.download(ticker, period="1y", progress=False, auto_adjust=True),
            max_attempts=2,
            sleep_seconds=_TOOL_RETRY_SLEEP_SECONDS,
        )
        if result.error:
            return result.format_failure(f"price history for {ticker}")
        hist = result.value
        if hist.empty:
            return f"No price history available for {ticker}."

        close = _select_history_series(hist, "Close", ticker)
        volume = _select_history_series(hist, "Volume", ticker)
        high = float(close.max())
        low = float(close.min())
        current = float(close.iloc[-1])
        start_of_year = close[close.index.year == close.index[-1].year].iloc[0]
        ytd_pct = (current - float(start_of_year)) / float(start_of_year) * 100
        avg_vol = float(volume.mean())

        lines = [
            f"Current Price:  ${current:.2f}",
            f"52-Week High:   ${high:.2f}",
            f"52-Week Low:    ${low:.2f}",
            f"YTD Return:     {ytd_pct:+.1f}%",
            f"Avg Daily Vol:  {avg_vol:,.0f} shares",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching price history for {ticker}: {e}"


def _select_history_series(hist, field: str, ticker: str):
    column = hist[field]
    if hasattr(column, "columns"):
        if ticker in column.columns:
            return column[ticker].squeeze()
        return column.iloc[:, 0].squeeze()
    return column.squeeze()
