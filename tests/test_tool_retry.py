import importlib
import sys
import unittest
from types import ModuleType

from tools.retry import RetryResult, retry_transient


def _install_fake_langchain_tool() -> None:
    core_tools = ModuleType("langchain_core.tools")

    class FakeStructuredTool:
        def __init__(self, func, name):
            self.func = func
            self.name = name

        def invoke(self, payload):
            return self.func(**payload)

    def fake_tool(name_or_func=None):
        if callable(name_or_func):
            return FakeStructuredTool(name_or_func, name_or_func.__name__)

        def decorator(func):
            return FakeStructuredTool(func, name_or_func or func.__name__)
        return decorator

    core_tools.tool = fake_tool
    sys.modules["langchain_core.tools"] = core_tools


class ToolRetryTests(unittest.TestCase):
    def test_retries_transient_timeout_then_returns_success_metadata(self):
        calls = {"count": 0}

        def flaky_call():
            calls["count"] += 1
            if calls["count"] == 1:
                raise TimeoutError("request timed out")
            return "ok"

        result = retry_transient(flaky_call, max_attempts=2, sleep_seconds=0)

        self.assertEqual(result.value, "ok")
        self.assertEqual(result.attempts, 2)
        self.assertTrue(result.retried)
        self.assertEqual(result.error_category, "")

    def test_does_not_retry_auth_errors(self):
        calls = {"count": 0}

        def auth_failure():
            calls["count"] += 1
            raise RuntimeError("401 Unauthorized")

        result = retry_transient(auth_failure, max_attempts=2, sleep_seconds=0)

        self.assertIsInstance(result.error, RuntimeError)
        self.assertEqual(result.attempts, 1)
        self.assertFalse(result.retried)
        self.assertEqual(result.error_category, "auth_error")

    def test_retry_result_formats_final_failure_for_tool_output(self):
        error = RuntimeError("server 500")
        result = RetryResult(
            value=None,
            error=error,
            attempts=2,
            retried=True,
            error_category="tool_error",
        )

        self.assertIn("after 2 attempts", result.format_failure("Tavily search"))
        self.assertIn("tool_error", result.format_failure("Tavily search"))

    def test_tavily_tool_retries_timeout_but_not_auth_error(self):
        community = ModuleType("langchain_community.tools.tavily_search")

        calls = {"count": 0}

        class FakeTavilySearchResults:
            def __init__(self, max_results):
                self.max_results = max_results

            def invoke(self, payload):
                calls["count"] += 1
                if calls["count"] == 1:
                    raise TimeoutError("request timed out")
                return [{"url": "https://example.com", "content": "ok"}]

        community.TavilySearchResults = FakeTavilySearchResults
        sys.modules["langchain_community.tools.tavily_search"] = community
        _install_fake_langchain_tool()
        sys.modules.pop("tools.tavily_tools", None)
        tavily_tools = importlib.import_module("tools.tavily_tools")

        try:
            tool = tavily_tools.get_tavily_tool(max_results=1, sleep_seconds=0)
            output = tool.invoke({"query": "Apple"})
        finally:
            sys.modules.pop("tools.tavily_tools", None)
            sys.modules.pop("langchain_community.tools.tavily_search", None)
            sys.modules.pop("langchain_core.tools", None)

        self.assertEqual(calls["count"], 2)
        self.assertIn("https://example.com", output)

        calls["count"] = 0

        class FakeAuthFailure:
            def __init__(self, max_results):
                self.max_results = max_results

            def invoke(self, payload):
                calls["count"] += 1
                raise RuntimeError("401 Unauthorized")

        community = ModuleType("langchain_community.tools.tavily_search")
        community.TavilySearchResults = FakeAuthFailure
        sys.modules["langchain_community.tools.tavily_search"] = community
        _install_fake_langchain_tool()
        tavily_tools = importlib.import_module("tools.tavily_tools")
        try:
            tool = tavily_tools.get_tavily_tool(max_results=1, sleep_seconds=0)
            output = tool.invoke({"query": "Apple"})
        finally:
            sys.modules.pop("tools.tavily_tools", None)
            sys.modules.pop("langchain_community.tools.tavily_search", None)
            sys.modules.pop("langchain_core.tools", None)

        self.assertEqual(calls["count"], 1)
        self.assertIn("auth_error", output)

    def test_yfinance_financial_summary_retries_transient_ticker_failure(self):
        yf_module = ModuleType("yfinance")
        calls = {"count": 0}

        class FakeTicker:
            @property
            def info(self):
                calls["count"] += 1
                if calls["count"] == 1:
                    raise TimeoutError("request timed out")
                return {
                    "longName": "Apple Inc.",
                    "sector": "Technology",
                    "industry": "Consumer Electronics",
                }

        yf_module.Ticker = lambda ticker: FakeTicker()
        yf_module.download = lambda *args, **kwargs: None
        sys.modules["yfinance"] = yf_module
        _install_fake_langchain_tool()
        sys.modules.pop("tools.yfinance_tools", None)

        try:
            yfinance_tools = importlib.import_module("tools.yfinance_tools")
            yfinance_tools._TOOL_RETRY_SLEEP_SECONDS = 0
            output = yfinance_tools.get_financial_summary.invoke({"ticker": "AAPL"})
        finally:
            sys.modules.pop("tools.yfinance_tools", None)
            sys.modules.pop("langchain_core.tools", None)
            sys.modules.pop("yfinance", None)

        self.assertEqual(calls["count"], 2)
        self.assertIn("Company: Apple Inc. (AAPL)", output)

    def test_yfinance_price_history_handles_single_ticker_multiindex_download(self):
        import pandas as pd

        yf_module = ModuleType("yfinance")
        dates = pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04"])
        yf_module.download = lambda *args, **kwargs: pd.DataFrame(
            {
                ("Close", "AAPL"): [100.0, 110.0, 120.0],
                ("Volume", "AAPL"): [1_000_000, 2_000_000, 3_000_000],
            },
            index=dates,
        )
        yf_module.Ticker = lambda ticker: None
        sys.modules["yfinance"] = yf_module
        _install_fake_langchain_tool()
        sys.modules.pop("tools.yfinance_tools", None)

        try:
            yfinance_tools = importlib.import_module("tools.yfinance_tools")
            yfinance_tools._TOOL_RETRY_SLEEP_SECONDS = 0
            output = yfinance_tools.get_price_history_summary.invoke({"ticker": "AAPL"})
        finally:
            sys.modules.pop("tools.yfinance_tools", None)
            sys.modules.pop("langchain_core.tools", None)
            sys.modules.pop("yfinance", None)

        self.assertIn("Current Price:  $120.00", output)
        self.assertIn("Avg Daily Vol:  2,000,000 shares", output)


if __name__ == "__main__":
    unittest.main()
