import importlib
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def _remove_modules(*names: str) -> None:
    for name in names:
        sys.modules.pop(name, None)


def _install_fake_genai() -> None:
    genai = ModuleType("langchain_google_genai")

    class FakeChatGoogleGenerativeAI:
        def __init__(self, *args, **kwargs):
            pass

        def with_fallbacks(self, fallbacks):
            return self

    genai.ChatGoogleGenerativeAI = FakeChatGoogleGenerativeAI
    sys.modules["langchain_google_genai"] = genai


def _install_fake_messages() -> None:
    messages = ModuleType("langchain_core.messages")

    class FakeMessage:
        def __init__(self, content):
            self.content = content

    messages.HumanMessage = FakeMessage
    messages.SystemMessage = FakeMessage
    sys.modules["langchain_core.messages"] = messages


def _install_fake_tool_modules() -> None:
    yfinance_tools = ModuleType("tools.yfinance_tools")
    yfinance_tools.get_financial_summary = lambda ticker: "summary"
    yfinance_tools.get_income_statement = lambda ticker: "income"
    yfinance_tools.get_price_history_summary = lambda ticker: "price"
    sys.modules["tools.yfinance_tools"] = yfinance_tools

    tavily_tools = ModuleType("tools.tavily_tools")
    tavily_tools.get_tavily_tool = lambda max_results=5: f"fake_tavily_{max_results}"
    sys.modules["tools.tavily_tools"] = tavily_tools


class FakeAgent:
    def __init__(self, content):
        self.content = content
        self.inputs = []

    def invoke(self, inputs):
        self.inputs.append(inputs)
        return {"messages": [SimpleNamespace(content=self.content)]}


class FakeTickerLLM:
    def __init__(self, content):
        self.content = content

    def invoke(self, messages):
        return SimpleNamespace(content=self.content)


class FakeReportLLM:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content="## 1. What Does This Company Do?\nReport body")


class Phase4HarnessTests(unittest.TestCase):
    def tearDown(self):
        _remove_modules(
            "agents.fundamental",
            "agents.business_model",
            "agents.news_sentiment",
            "agents.report_writer",
            "graph",
            "langchain_google_genai",
            "langchain_core.messages",
            "langgraph.prebuilt",
            "langgraph.graph",
            "tools.yfinance_tools",
            "tools.tavily_tools",
        )

    def test_graph_routing_uses_mocked_ticker_resolver_without_agent_calls(self):
        _install_fake_genai()
        _install_fake_messages()
        for module_name, attr in [
            ("agents.business_model", "business_model_node"),
            ("agents.fundamental", "fundamental_node"),
            ("agents.news_sentiment", "news_sentiment_node"),
            ("agents.report_writer", "report_writer_node"),
        ]:
            module = ModuleType(module_name)
            setattr(module, attr, lambda state: {})
            sys.modules[module_name] = module

        graph_module = ModuleType("langgraph.graph")
        graph_module.END = "__end__"
        graph_module.START = "__start__"
        graph_module.StateGraph = object
        sys.modules["langgraph.graph"] = graph_module

        graph = importlib.import_module("graph")
        graph._ticker_llm = FakeTickerLLM("AAPL")

        resolved = graph.orchestrate_node({"company_input": "Apple", "report_cache": {}})
        self.assertEqual(resolved["ticker"], "AAPL")
        self.assertEqual(
            graph.route_to_agents(resolved),
            ["fundamental", "business_model", "news_sentiment"],
        )

        graph._ticker_llm = FakeTickerLLM("UNKNOWN")
        unknown = graph.orchestrate_node({"company_input": "not a company", "report_cache": {}})
        self.assertIn("Could not identify a stock ticker", unknown["error"])
        self.assertEqual(graph.route_to_agents(unknown), graph.END)

        graph._ticker_llm = FakeTickerLLM("AAPL")
        cached = graph.orchestrate_node({
            "company_input": "Apple",
            "report_cache": {"AAPL": "cached report"},
        })
        self.assertTrue(cached["cache_hit"])
        self.assertEqual(cached["report"], "cached report")
        self.assertEqual(graph.route_to_agents(cached), graph.END)

    def test_specialist_nodes_parse_mocked_agent_json_without_tool_network_calls(self):
        _install_fake_genai()
        _install_fake_tool_modules()
        prebuilt = ModuleType("langgraph.prebuilt")
        prebuilt.create_react_agent = lambda *args, **kwargs: FakeAgent("{}")
        sys.modules["langgraph.prebuilt"] = prebuilt

        content = json.dumps({
            "summary": "Structured summary",
            "key_facts": ["Fact"],
            "risks": ["Risk"],
            "sources": ["Source"],
            "data_quality": "good",
            "confidence": "high",
        })

        fundamental = importlib.import_module("agents.fundamental")
        business = importlib.import_module("agents.business_model")
        news = importlib.import_module("agents.news_sentiment")

        fundamental._agent = FakeAgent(content)
        business._agent = FakeAgent(content)
        news._agent = FakeAgent(content)

        self.assertEqual(
            fundamental.fundamental_node({"ticker": "AAPL"})["fundamental_data"]["summary"],
            "Structured summary",
        )
        self.assertEqual(
            business.business_model_node({"ticker": "AAPL", "company_input": "Apple"})["business_model_data"]["sources"],
            ["Source"],
        )
        self.assertEqual(
            news.news_sentiment_node({"ticker": "AAPL", "company_input": "Apple"})["news_data"]["data_quality"],
            "good",
        )

    def test_specialist_node_tool_failure_text_becomes_weak_structured_context(self):
        _install_fake_genai()
        _install_fake_tool_modules()
        prebuilt = ModuleType("langgraph.prebuilt")
        prebuilt.create_react_agent = lambda *args, **kwargs: FakeAgent("{}")
        sys.modules["langgraph.prebuilt"] = prebuilt

        fundamental = importlib.import_module("agents.fundamental")
        fundamental._agent = FakeAgent("Error fetching financial summary for SPARSE: no data")

        result = fundamental.fundamental_node({"ticker": "SPARSE"})["fundamental_data"]

        self.assertEqual(result["data_quality"], "weak")
        self.assertEqual(result["confidence"], "low")
        self.assertEqual(result["key_facts"], [])
        self.assertIn("Error fetching financial summary", result["summary"])

    def test_report_writer_prompt_contains_structured_context_without_real_llm_call(self):
        _install_fake_genai()
        _install_fake_messages()
        report_writer = importlib.import_module("agents.report_writer")
        fake_llm = FakeReportLLM()
        report_writer._llm = fake_llm

        output = {
            "summary": "Strong business.",
            "key_facts": ["Fact with number."],
            "risks": ["Important risk."],
            "sources": ["https://example.com"],
            "data_quality": "partial",
            "confidence": "medium",
            "raw_text": "raw context",
        }

        result = report_writer.report_writer_node({
            "company_input": "Apple",
            "ticker": "AAPL",
            "fundamental_data": output,
            "business_model_data": output,
            "news_data": output,
        })

        prompt = fake_llm.messages[-1].content
        self.assertIn("Data quality: partial", prompt)
        self.assertIn("Confidence: medium", prompt)
        self.assertIn("Key facts:\n- Fact with number.", prompt)
        self.assertIn("Risks/Caveats:\n- Important risk.", prompt)
        self.assertIn("Sources:\n- https://example.com", prompt)
        self.assertTrue(result["report"].startswith("## 1. What Does This Company Do?"))

    def test_evaluation_cases_document_representative_expected_traits(self):
        cases = json.loads((ROOT / "evals" / "cases.json").read_text(encoding="utf-8"))

        names = {case["name"] for case in cases}
        self.assertEqual(
            names,
            {"apple", "tencent", "unknown_company", "sparse_data_ticker"},
        )

        for case in cases:
            self.assertIn("input", case)
            self.assertIn("expected_traits", case)
            self.assertGreaterEqual(len(case["expected_traits"]), 3)
