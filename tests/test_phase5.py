import importlib
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

        def invoke(self, messages):
            return SimpleNamespace(content="primary report")

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


class FakeLLM:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.content)


class Phase5SourceAndObservabilityTests(unittest.TestCase):
    def tearDown(self):
        _remove_modules(
            "agents.report_writer",
            "agents.structured_output",
            "observability",
            "source_quality",
            "langchain_google_genai",
            "langchain_core.messages",
        )

    def test_rank_sources_deduplicates_and_prefers_traceable_urls(self):
        source_quality = importlib.import_module("source_quality")

        ranked = source_quality.rank_sources([
            "N/A",
            "Company annual report",
            "https://example.com/article",
            "https://www.sec.gov/Archives/example",
            "https://example.com/article",
            "",
        ])

        self.assertEqual(ranked[0]["source"], "https://www.sec.gov/Archives/example")
        self.assertEqual(ranked[0]["traceability"], "url")
        self.assertEqual([item["source"] for item in ranked].count("https://example.com/article"), 1)
        self.assertNotIn("N/A", [item["source"] for item in ranked])

    def test_specialist_context_includes_source_quality_and_traceable_sources(self):
        structured = importlib.import_module("agents.structured_output")

        context = structured.format_specialist_context(
            "News",
            {
                "summary": "Recent earnings beat expectations.",
                "key_facts": ["Earnings beat was reported by Example News."],
                "risks": ["Regulatory pressure remains."],
                "sources": ["Example News", "https://example.com/news"],
                "data_quality": "good",
                "confidence": "high",
                "raw_text": "raw",
            },
        )

        self.assertIn("Source quality:", context)
        self.assertIn("Traceable sources:", context)
        self.assertIn("- https://example.com/news", context)
        self.assertIn("Use source-backed claims first", context)

    def test_trace_node_records_success_and_failure_events(self):
        observability = importlib.import_module("observability")

        def success_node(state):
            return {"value": 1}

        def failing_node(state):
            raise RuntimeError("boom")

        success = observability.trace_node("success", success_node, fallback_output={"value": 0})
        failure = observability.trace_node("failure", failing_node, fallback_output={"value": 0})

        success_result = success({})
        failure_result = failure({})

        self.assertEqual(success_result["value"], 1)
        self.assertEqual(success_result["run_events"][0]["node"], "success")
        self.assertEqual(success_result["run_events"][0]["status"], "success")
        self.assertIsInstance(success_result["run_events"][0]["latency_ms"], int)
        self.assertEqual(failure_result["value"], 0)
        self.assertEqual(failure_result["run_events"][0]["status"], "error")
        self.assertIn("boom", failure_result["run_events"][0]["error"])

    def test_tool_events_from_react_messages_summarize_tool_activity(self):
        observability = importlib.import_module("observability")

        result = {
            "messages": [
                SimpleNamespace(type="human", content="question"),
                SimpleNamespace(type="ai", tool_calls=[{"name": "search"}, {"name": "search"}]),
                SimpleNamespace(type="tool", name="search", content="tool output"),
            ]
        }

        events = observability.tool_events_from_react_result("news_sentiment", result)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["node"], "news_sentiment.tools")
        self.assertEqual(events[0]["status"], "success")
        self.assertIn("2 requested, 1 completed", events[0]["error"])
        self.assertEqual(events[0]["error_category"], "")

    def test_tool_events_include_successful_retry_notes(self):
        observability = importlib.import_module("observability")

        result = {
            "messages": [
                SimpleNamespace(type="ai", tool_calls=[{"name": "tavily_search_results_json"}]),
                SimpleNamespace(
                    type="tool",
                    name="tavily_search_results_json",
                    content="Tool retry note: succeeded after 2 attempts.\n[]",
                ),
            ]
        }

        events = observability.tool_events_from_react_result("business_model", result)

        self.assertEqual(events[0]["status"], "success")
        self.assertIn("succeeded after 2 attempts", events[0]["error"])
        self.assertEqual(events[0]["error_category"], "")

    def test_tool_events_mark_wrapped_http_errors_as_failed_tool_activity(self):
        observability = importlib.import_module("observability")

        result = {
            "messages": [
                SimpleNamespace(type="ai", tool_calls=[{"name": "tavily_search_results_json"}]),
                SimpleNamespace(
                    type="tool",
                    name="tavily_search_results_json",
                    content="HTTPError('401 Client Error: Unauthorized for url: https://api.tavily.com/search')",
                ),
            ]
        }

        events = observability.tool_events_from_react_result("business_model", result)

        self.assertEqual(events[0]["status"], "error")
        self.assertIn("Unauthorized", events[0]["error"])
        self.assertEqual(events[0]["error_category"], "auth_error")

    def test_error_classification_covers_common_failure_types(self):
        observability = importlib.import_module("observability")

        self.assertEqual(observability.classify_error("401 Unauthorized"), "auth_error")
        self.assertEqual(observability.classify_error("429 rate limit exceeded"), "rate_limit")
        self.assertEqual(observability.classify_error("JSONDecodeError malformed"), "parse_error")
        self.assertEqual(observability.classify_error("Read timeout"), "timeout")
        self.assertEqual(observability.classify_error("model unavailable"), "model_error")

    def test_tool_error_downgrades_specialist_output_quality(self):
        structured = importlib.import_module("agents.structured_output")

        output = {
            "summary": "Search-backed summary.",
            "key_facts": ["Fact"],
            "risks": [],
            "sources": [],
            "data_quality": "good",
            "confidence": "high",
            "raw_text": "{}",
        }
        downgraded = structured.apply_tool_error_context(output, [
            {
                "node": "business_model.tools",
                "status": "error",
                "latency_ms": 0,
                "error": "HTTPError 401 Unauthorized",
                "model": "",
                "fallback_model": "",
                "fallback_used": False,
            }
        ])

        self.assertEqual(downgraded["data_quality"], "weak")
        self.assertEqual(downgraded["confidence"], "low")
        self.assertIn("Tool failure", downgraded["risks"][0])

    def test_report_writer_records_fallback_model_and_includes_run_events_in_prompt(self):
        _install_fake_genai()
        _install_fake_messages()
        report_writer = importlib.import_module("agents.report_writer")

        report_writer._primary = FakeLLM(error=RuntimeError("primary failed"))
        report_writer._fallback = FakeLLM(content="fallback report")

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
            "run_events": [
                {
                    "node": "news_sentiment",
                    "status": "success",
                    "latency_ms": 1234,
                    "error": "",
                    "model": "gemini-3.1-flash-lite",
                    "fallback_model": "gemini-2.5-flash-lite",
                    "fallback_used": False,
                }
            ],
        })

        prompt = report_writer._fallback.messages[-1].content
        self.assertEqual(result["report"], "fallback report")
        self.assertEqual(result["run_events"][0]["node"], "report_writer")
        self.assertTrue(result["run_events"][0]["fallback_used"])
        self.assertEqual(result["run_events"][0]["model"], "gemini-2.5-flash-lite")
        self.assertIn("RUN OBSERVABILITY", prompt)
        self.assertIn("news_sentiment: success in 1234 ms", prompt)

    def test_report_writer_returns_emergency_five_section_report_when_both_models_fail(self):
        _install_fake_genai()
        _install_fake_messages()
        report_writer = importlib.import_module("agents.report_writer")

        report_writer._primary = FakeLLM(error=RuntimeError("primary model unavailable"))
        report_writer._fallback = FakeLLM(error=RuntimeError("fallback model unavailable"))

        result = report_writer.report_writer_node({
            "company_input": "Apple",
            "ticker": "AAPL",
            "fundamental_data": None,
            "business_model_data": None,
            "news_data": None,
            "run_events": [],
        })

        self.assertIn("## 1. What Does This Company Do?", result["report"])
        self.assertIn("## 5. One-Line Verdict", result["report"])
        self.assertIn("Data note:", result["report"])
        self.assertEqual(result["run_events"][0]["status"], "error")
        self.assertEqual(result["run_events"][0]["error_category"], "model_error")

    def test_report_writer_prompt_groups_low_confidence_caveats_at_report_end(self):
        _install_fake_genai()
        _install_fake_messages()
        report_writer = importlib.import_module("agents.report_writer")

        fake_llm = FakeLLM(content="report")
        report_writer._primary = fake_llm

        good_output = {
            "summary": "Strong business.",
            "key_facts": ["Fact with number."],
            "risks": [],
            "sources": ["https://example.com"],
            "data_quality": "good",
            "confidence": "high",
            "raw_text": "raw context",
        }
        weak_news = {
            "summary": "News summary from weak context.",
            "key_facts": ["Unsourced news fact."],
            "risks": ["Tool failure limited this analysis: HTTPError 401 Unauthorized"],
            "sources": [],
            "data_quality": "weak",
            "confidence": "low",
            "raw_text": "raw weak context",
        }

        report_writer.report_writer_node({
            "company_input": "Nvidia",
            "ticker": "NVDA",
            "fundamental_data": good_output,
            "business_model_data": good_output,
            "news_data": weak_news,
            "run_events": [],
        })

        prompt = fake_llm.messages[-1].content
        self.assertIn("REQUIRED UNCERTAINTY NOTES", prompt)
        self.assertIn("News & sentiment", prompt)
        self.assertIn("Group these notes into one concise Data note at the end of section 5", prompt)
        self.assertIn("Do not repeat these caveats throughout sections 1-4", prompt)
        self.assertNotIn("HTTPError", prompt.split("--- REQUIRED UNCERTAINTY NOTES ---", 1)[1])

    def test_cli_warning_summarizes_weak_specialist_data(self):
        main = importlib.import_module("main")

        warning = main.summarize_data_warning({
            "fundamental_data": {
                "data_quality": "good",
                "confidence": "high",
            },
            "business_model_data": {
                "data_quality": "weak",
                "confidence": "low",
            },
            "news_data": {
                "data_quality": "partial",
                "confidence": "medium",
            },
        })

        self.assertEqual(
            warning,
            "[!] Business model data was weak; report includes a data note.",
        )


if __name__ == "__main__":
    unittest.main()
