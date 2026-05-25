import os
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
    on_update: Optional[Callable[[str, dict], None]] = None,
) -> dict:
    """Run one query and update the session cache only after fresh reports."""
    result: dict = {}
    initial_state = {"company_input": user_input, "report_cache": cache}

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
        return result

    report = result.get("report", "No report generated.")
    if ticker:
        cache[ticker] = report
    result["report"] = report
    result["cache_hit"] = False
    return result


def main() -> int:
    try:
        validate_required_environment()
    except RuntimeError as exc:
        print(f"Startup error: {exc}")
        return 1

    from graph import build_graph  # env must be validated before graph imports agents

    graph = build_graph()
    cache: dict[str, str] = {}  # ticker -> report, lives for this session

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
        result = process_query(graph, user_input, cache, on_update=print_progress)
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

        print(result.get("report", "No report generated."))
        print(_DIVIDER)


if __name__ == "__main__":
    raise SystemExit(main())
