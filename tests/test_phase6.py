import importlib
import json
import sys
import unittest
from types import ModuleType, SimpleNamespace


def _remove_modules(*names: str) -> None:
    for name in names:
        sys.modules.pop(name, None)


def _install_fake_genai() -> None:
    genai = ModuleType("langchain_google_genai")

    class FakeChatGoogleGenerativeAI:
        def __init__(self, *args, **kwargs):
            pass

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
    def __init__(self, content):
        self.content = content
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=self.content)


class FakeTickerLLM:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return SimpleNamespace(content=self.content)


class FakeStreamGraph:
    def __init__(self, updates):
        self.updates = updates

    def stream(self, state, stream_mode):
        yield from self.updates


class Phase6PlannerMemoryTests(unittest.TestCase):
    def tearDown(self):
        _remove_modules(
            "agents.planner",
            "agents.report_writer",
            "agents.business_model",
            "agents.fundamental",
            "agents.news_sentiment",
            "graph",
            "langchain_google_genai",
            "langchain_core.messages",
            "langgraph.graph",
        )

    def test_planner_parses_structured_plan_with_rewritten_task(self):
        _install_fake_genai()
        _install_fake_messages()
        planner = importlib.import_module("agents.planner")
        planner._planner_llm = FakeLLM(json.dumps({
            "intent": "business_model_only",
            "target_company": "Google",
            "reuse_last_ticker": False,
            "required_agents": ["business_model"],
            "answer_style": "short_answer",
            "rewritten_task": "Explain Alphabet's business model only.",
        }))

        result = planner.planner_node({
            "company_input": "how does Google make money?",
            "session_memory": {},
        })

        plan = result["planner"]
        self.assertEqual(plan["intent"], "business_model_only")
        self.assertEqual(plan["required_agents"], ["business_model"])
        self.assertEqual(plan["answer_style"], "short_answer")
        self.assertIn("business model", plan["rewritten_task"])

    def test_planner_prompt_prefers_minimal_agents_for_narrow_queries(self):
        _install_fake_genai()
        _install_fake_messages()
        planner = importlib.import_module("agents.planner")
        fake_llm = FakeLLM(json.dumps({
            "intent": "financial_only",
            "target_company": "NVIDIA",
            "reuse_last_ticker": False,
            "required_agents": ["fundamental"],
            "answer_style": "short_answer",
            "rewritten_task": "Assess whether NVIDIA looks expensive.",
        }))
        planner._planner_llm = fake_llm

        planner.planner_node({
            "company_input": "is nvidia expensive?",
            "session_memory": {},
        })

        prompt_text = "\n".join(message.content for message in fake_llm.messages)
        self.assertIn("Use the fewest specialist agents", prompt_text)
        self.assertIn("financial_only", prompt_text)
        self.assertIn('["fundamental"]', prompt_text)
        self.assertIn("short_answer", prompt_text)

    def test_planner_prompt_reuses_last_ticker_when_company_is_omitted(self):
        _install_fake_genai()
        _install_fake_messages()
        planner = importlib.import_module("agents.planner")
        fake_llm = FakeLLM(json.dumps({
            "intent": "business_model_only",
            "target_company": "",
            "reuse_last_ticker": True,
            "required_agents": ["business_model"],
            "answer_style": "short_answer",
            "rewritten_task": "Explain the supplier base for the remembered company.",
        }))
        planner._planner_llm = fake_llm

        planner.planner_node({
            "company_input": "who are the suppliers?",
            "session_memory": {"last_ticker": "GOOGL", "last_company_input": "Google"},
        })

        prompt_text = "\n".join(message.content for message in fake_llm.messages)
        self.assertIn("If the user omits a company", prompt_text)
        self.assertIn("reuse_last_ticker", prompt_text)
        self.assertIn("last_ticker", prompt_text)

    def test_planner_fallback_reuses_last_ticker_for_follow_up(self):
        _install_fake_genai()
        _install_fake_messages()
        planner = importlib.import_module("agents.planner")
        planner._planner_llm = FakeLLM("not json")

        result = planner.planner_node({
            "company_input": "what are the risks?",
            "session_memory": {"last_ticker": "AAPL", "last_report": "old report"},
        })

        plan = result["planner"]
        self.assertTrue(plan["reuse_last_ticker"])
        self.assertEqual(plan["target_company"], "")
        self.assertEqual(plan["required_agents"], [])
        self.assertEqual(plan["answer_style"], "short_answer")

    def test_bare_company_name_guardrail_keeps_full_report_cache_path(self):
        _install_fake_genai()
        _install_fake_messages()
        planner = importlib.import_module("agents.planner")
        planner._planner_llm = FakeLLM(json.dumps({
            "intent": "news_only",
            "target_company": "Apple",
            "reuse_last_ticker": False,
            "required_agents": ["news_sentiment"],
            "answer_style": "short_answer",
            "rewritten_task": "Summarize recent Apple news.",
        }))

        result = planner.planner_node({
            "company_input": "apple",
            "session_memory": {"last_ticker": "AAPL", "last_company_input": "Apple"},
        })

        plan = result["planner"]
        self.assertEqual(plan["intent"], "full_report")
        self.assertEqual(plan["target_company"], "apple")
        self.assertFalse(plan["reuse_last_ticker"])
        self.assertEqual(plan["required_agents"], ["fundamental", "business_model", "news_sentiment"])
        self.assertEqual(plan["answer_style"], "five_section_report")

    def test_orchestrate_reuses_memory_ticker_for_follow_up_without_ticker_llm(self):
        _install_fake_genai()
        _install_fake_messages()
        for module_name, attr in [
            ("agents.planner", "planner_node"),
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
        fake_ticker_llm = FakeTickerLLM("SHOULD_NOT_BE_USED")
        graph._ticker_llm = fake_ticker_llm

        result = graph.orchestrate_node({
            "company_input": "what are the risks?",
            "planner": {
                "reuse_last_ticker": True,
                "required_agents": [],
                "answer_style": "short_answer",
            },
            "session_memory": {
                "last_ticker": "AAPL",
                "last_report": "previous Apple report",
            },
            "report_cache": {"AAPL": "cached Apple report"},
        })

        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["previous_report"], "cached Apple report")
        self.assertEqual(fake_ticker_llm.calls, 0)
        self.assertEqual(graph.route_to_agents(result), "report_writer")

    def test_route_to_agents_uses_planner_required_agents(self):
        _install_fake_genai()
        _install_fake_messages()
        for module_name, attr in [
            ("agents.planner", "planner_node"),
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

        route = graph.route_to_agents({
            "ticker": "GOOGL",
            "planner": {"required_agents": ["business_model"], "answer_style": "short_answer"},
        })

        self.assertEqual(route, ["business_model"])

    def test_session_memory_updates_after_successful_report(self):
        import main

        memory = main.SessionMemory()
        main.update_session_memory(
            memory,
            {
                "ticker": "NVDA",
                "report": "Nvidia report",
                "planner": {"intent": "full_report"},
            },
            "Nvidia",
        )

        self.assertEqual(memory.last_ticker, "NVDA")
        self.assertEqual(memory.last_report, "Nvidia report")
        self.assertEqual(memory.recent_queries[-1]["intent"], "full_report")

    def test_short_answers_do_not_overwrite_full_report_cache(self):
        import main

        cache = {"GOOGL": "full Google report"}
        graph = FakeStreamGraph([
            {
                "report_writer": {
                    "ticker": "GOOGL",
                    "report": "short business model answer",
                    "planner": {"answer_style": "short_answer", "intent": "business_model_only"},
                    "error": None,
                }
            }
        ])

        result = main.process_query(graph, "how does Google make money?", cache)

        self.assertFalse(result["cache_hit"])
        self.assertEqual(result["report"], "short business model answer")
        self.assertEqual(cache["GOOGL"], "full Google report")

    def test_short_answer_uncertainty_notes_do_not_reference_section_five(self):
        _install_fake_genai()
        _install_fake_messages()
        report_writer = importlib.import_module("agents.report_writer")

        output = {
            "summary": "Business model context.",
            "key_facts": [],
            "risks": [],
            "sources": [],
            "data_quality": "weak",
            "confidence": "low",
            "raw_text": "raw",
        }

        notes = report_writer._format_uncertainty_notes({
            "planner": {
                "answer_style": "short_answer",
                "required_agents": ["business_model"],
            },
            "business_model_data": output,
        })

        self.assertIn("end of the answer", notes)
        self.assertNotIn("section 5", notes)

    def test_short_answer_prompt_omits_skipped_specialist_context(self):
        _install_fake_genai()
        _install_fake_messages()
        report_writer = importlib.import_module("agents.report_writer")

        fake_llm = FakeLLM("risk answer")
        report_writer._primary = fake_llm

        output = {
            "summary": "Financial context.",
            "key_facts": ["Fact"],
            "risks": [],
            "sources": ["yfinance"],
            "data_quality": "good",
            "confidence": "high",
            "raw_text": "raw",
        }

        report_writer.report_writer_node({
            "company_input": "what are the main risks?",
            "ticker": "AAPL",
            "previous_report": "Previous Apple report",
            "planner": {
                "intent": "risk_followup",
                "required_agents": ["fundamental"],
                "answer_style": "short_answer",
                "rewritten_task": "Answer the user's risk follow-up.",
            },
            "fundamental_data": output,
            "business_model_data": None,
            "news_data": None,
            "run_events": [],
        })

        prompt = fake_llm.messages[-1].content
        self.assertIn("--- FUNDAMENTAL DATA ---", prompt)
        self.assertNotIn("--- BUSINESS MODEL DATA ---", prompt)
        self.assertNotIn("--- NEWS & SENTIMENT DATA ---", prompt)
        self.assertNotIn("No structured context was available", prompt)

    def test_short_answer_with_good_named_sources_tells_writer_not_to_add_data_note(self):
        _install_fake_genai()
        _install_fake_messages()
        report_writer = importlib.import_module("agents.report_writer")

        fake_llm = FakeLLM("risk answer")
        report_writer._primary = fake_llm

        output = {
            "summary": "Risk context.",
            "key_facts": ["Regulatory risk remains important."],
            "risks": ["Antitrust pressure."],
            "sources": ["yfinance", "Reuters", "Bloomberg"],
            "data_quality": "good",
            "confidence": "high",
            "raw_text": "raw",
        }

        report_writer.report_writer_node({
            "company_input": "what are the main risks?",
            "ticker": "AAPL",
            "previous_report": "Previous Apple report",
            "planner": {
                "intent": "risk_followup",
                "required_agents": ["fundamental"],
                "answer_style": "short_answer",
                "rewritten_task": "Answer the user's risk follow-up.",
            },
            "fundamental_data": output,
            "run_events": [],
        })

        prompt = fake_llm.messages[-1].content
        uncertainty_block = prompt.split("--- REQUIRED UNCERTAINTY NOTES ---", 1)[1]
        self.assertIn("None. Do not add a Data note", uncertainty_block)
        self.assertIn("do not treat missing URLs alone as weak evidence", prompt)
        self.assertNotIn("No traceable source URLs were available", uncertainty_block)


if __name__ == "__main__":
    unittest.main()
