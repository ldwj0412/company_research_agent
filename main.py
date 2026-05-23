from dotenv import load_dotenv

load_dotenv()

from graph import build_graph  # noqa: E402 — must load env before importing agents

_DIVIDER = "\n" + "─" * 60 + "\n"

_NODE_LABELS = {
    "orchestrate":    "Ticker resolved",
    "fundamental":    "Financials done",
    "business_model": "Business model done",
    "news_sentiment": "News & sentiment done",
    "report_writer":  "Report ready",
}


def main():
    graph = build_graph()
    _cache: dict[str, str] = {}  # ticker → report, lives for this session

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
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        # Stream node-by-node so the user sees progress as each agent finishes
        result: dict = {}
        print()
        for update in graph.stream({"company_input": user_input}, stream_mode="updates"):
            for node_name, node_output in update.items():
                result.update(node_output)
                label = _NODE_LABELS.get(node_name, node_name)
                ticker_hint = f" ({result.get('ticker', '')})" if node_name == "orchestrate" else ""
                print(f"  [+] {label}{ticker_hint}")

        print()

        if result.get("error"):
            print(f"Sorry, I couldn't process that: {result['error']}\n")
            continue

        ticker = result.get("ticker", "")

        # Cache hit — same company asked again this session
        if ticker and ticker in _cache:
            print(f"[Showing cached report for {ticker}]\n")
            print(_cache[ticker])
            print(_DIVIDER)
            continue

        report = result.get("report", "No report generated.")
        if ticker:
            _cache[ticker] = report

        print(report)
        print(_DIVIDER)


if __name__ == "__main__":
    main()
