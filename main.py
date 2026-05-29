import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from dotenv import load_dotenv

load_dotenv()

REQUIRED_ENV_VARS = ("GOOGLE_API_KEY", "TAVILY_API_KEY")

_DIVIDER = "\n" + "─" * 60 + "\n"

_NODE_LABELS = {
    "orchestrate": "Ticker resolved",
    "fundamental": "Financials done",
    "business_model": "Business model done",
    "news_sentiment": "News & sentiment done",
    "report_writer": "Report ready",
}

_DATA_LABELS = {
    "fundamental_data": "Financial data",
    "business_model_data": "Business model data",
    "news_data": "News data",
}


@dataclass
class SessionMemory:
    last_ticker: Optional[str] = None
    last_company_input: Optional[str] = None
    last_report: Optional[str] = None
    recent_queries: list[dict] = field(default_factory=list)

    def to_state(self) -> dict:
        return {
            "last_ticker": self.last_ticker,
            "last_company_input": self.last_company_input,
            "last_report": self.last_report,
            "recent_queries": self.recent_queries[-5:],
        }


def validate_required_environment(environ: Optional[dict[str, str]] = None) -> None:
    """Fail fast when credentials needed by the graph are not configured."""
    values = os.environ if environ is None else environ
    missing = [name for name in REQUIRED_ENV_VARS if not values.get(name)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing required environment variables: {joined}. "
            "Set them in .env or your shell before starting."
        )


def process_query(
    graph,
    user_input: str,
    cache: dict[str, str],
    memory: Optional[SessionMemory] = None,
    on_update: Optional[Callable[[str, dict], None]] = None,
) -> dict:
    """Run one query and update the session cache only after fresh reports."""
    result: dict = {}
    initial_state = {
        "company_input": user_input,
        "report_cache": cache,
        "session_memory": memory.to_state() if memory else {},
    }

    for update in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in update.items():
            result.update(node_output)
            if on_update:
                on_update(node_name, result)

    if result.get("error"):
        result["cache_hit"] = False
        return result

    ticker = result.get("ticker", "")
    inferred_cache_hit = bool(ticker and ticker in cache and result.get("report") == cache[ticker])
    if result.get("cache_hit") or inferred_cache_hit:
        result["cache_hit"] = True
        if memory:
            update_session_memory(memory, result, user_input)
        return result

    report = result.get("report", "No report generated.")
    planner = result.get("planner") or {}
    if ticker and planner.get("answer_style", "five_section_report") == "five_section_report":
        cache[ticker] = report
    result["report"] = report
    result["cache_hit"] = False
    if memory:
        update_session_memory(memory, result, user_input)
    return result


def update_session_memory(memory: SessionMemory, result: dict, user_input: str) -> None:
    if result.get("error"):
        return

    ticker = result.get("ticker")
    report = result.get("report")
    if ticker:
        memory.last_ticker = ticker
    if report:
        memory.last_report = report
    memory.last_company_input = user_input
    memory.recent_queries.append({
        "input": user_input,
        "ticker": ticker or "",
        "intent": (result.get("planner") or {}).get("intent", ""),
    })
    del memory.recent_queries[:-5]


def summarize_data_warning(result: dict) -> str:
    weak_labels = []
    for key, label in _DATA_LABELS.items():
        output = result.get(key) or {}
        if output.get("data_quality") == "weak" or output.get("confidence") == "low":
            weak_labels.append(label)

    if not weak_labels:
        return ""

    if len(weak_labels) == 1:
        subject = weak_labels[0]
        verb = "was"
    else:
        subject = ", ".join(weak_labels[:-1]) + f" and {weak_labels[-1]}"
        verb = "were"
    return f"[!] {subject} {verb} weak; report includes a data note."


def main() -> int:
    try:
        validate_required_environment()
    except RuntimeError as exc:
        print(f"Startup error: {exc}")
        return 1

    from graph import build_graph  # env must be validated before graph imports agents

    graph = build_graph()
    cache: dict[str, str] = {}  # ticker -> report, lives for this session
    memory = SessionMemory()

    print("=" * 60)
    print("  Company Research Agent")
    print("  Ask me about any company. Type 'quit' to exit.")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            return 0

        def print_progress(node_name: str, result: dict) -> None:
            label = _NODE_LABELS.get(node_name, node_name)
            ticker_hint = f" ({result.get('ticker', '')})" if node_name == "orchestrate" else ""
            print(f"  [+] {label}{ticker_hint}")

        # Stream node-by-node so the user sees progress as each agent finishes.
        print()
        result = process_query(graph, user_input, cache, memory=memory, on_update=print_progress)
        print()

        if result.get("error"):
            print(f"Sorry, I couldn't process that: {result['error']}\n")
            continue

        ticker = result.get("ticker", "")
        if result.get("cache_hit"):
            print(f"[Showing cached report for {ticker}]\n")
            print(result.get("report", "No report generated."))
            print(_DIVIDER)
            continue

        warning = summarize_data_warning(result)
        if warning:
            print(f"{warning}\n")

        print(result.get("report", "No report generated."))
        print(_DIVIDER)


if __name__ == "__main__":
    raise SystemExit(main())
