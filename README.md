# Company Research Agent

Company Research Agent is a Python CLI research assistant for HK and US listed companies. It accepts a company name, ticker, or natural-language request and returns a plain-English equity research brief covering the business model, financial health, recent news, risks, and a one-line verdict.

This is an AI-engineering portfolio project focused on multi-agent orchestration, tool use, and context handoff between specialist agents.

## What It Does

- Resolves natural-language company input to a listed stock ticker.
- Runs three specialist research agents in parallel:
  - Fundamentals via yfinance tools.
  - Business model and competitive position via Tavily search.
  - Recent news, sentiment, regulatory, and macro context via Tavily search.
- Synthesizes the specialist outputs into a five-section equity research report.
- Streams node-level progress so the user can see which research stage has finished.
- Uses an in-memory session cache so repeated queries for the same resolved ticker avoid re-running the expensive research agents.

## Architecture

Current runtime topology:

```text
                    +--------------+
                    |   User CLI   |
                    +------+-------+
                           |
                           v
                    +--------------+
                    |   Planner    |
                    | decide intent|
                    | choose agents|
                    +------+-------+
                           |
                           v
                    +--------------+
                    | Orchestrator |
                    |resolve ticker|
                    | check cache  |
                    +------+-------+
                           |
              +------------+------------+
              |            |            |
              v            v            v
       +------------+ +------------+ +------------+
       | Financials | | Business   | | News       |
       | yfinance   | | Tavily     | | Tavily     |
       +-----+------+ +-----+------+ +-----+------+
             |              |              |
             +--------------+--------------+
                            |
                            v
                    +--------------+
                    |Report Writer |
                    | synthesize   |
                    +------+-------+
                           |
                           v
                    +--------------+
                    | Final Answer |
                    +--------------+
```

Main flow:

1. `main.py` starts the CLI, validates required environment variables, streams graph updates, and keeps session memory plus the report cache.
2. `agents/planner.py` classifies the user's intent, chooses the smallest sufficient specialist-agent set, and decides whether the answer should be a short answer or a five-section report.
3. `graph.py` resolves the ticker, reuses the last ticker for follow-ups when the planner requests it, checks full-report cache hits, and routes to the selected specialists.
4. `agents/fundamental.py` uses yfinance tools to summarize financial health.
5. `agents/business_model.py` uses Tavily search to research revenue model, moat, competitors, and geographic exposure.
6. `agents/news_sentiment.py` uses Tavily search to summarize recent news, management changes, regulation, macro context, and sentiment.
7. `agents/report_writer.py` makes a direct LLM call to synthesize the available research outputs into either a short answer or the final five-section report.

The specialist agents are ReAct/tool-using agents. The planner and report writer are direct LLM calls, which keeps routing and synthesis predictable.

## Project Structure

```text
Company research agent/
|-- main.py
|-- graph.py
|-- state.py
|-- requirements.txt
|-- README.md
|-- ROADMAP.md
|-- agents/
|   |-- planner.py
|   |-- fundamental.py
|   |-- business_model.py
|   |-- news_sentiment.py
|   |-- structured_output.py
|   `-- report_writer.py
|-- tools/
|   |-- yfinance_tools.py
|   `-- tavily_tools.py
`-- prompts/
    |-- fundamental.txt
    |-- business_model.txt
    |-- news_sentiment.txt
    `-- report_writer.txt
```

## Tech Stack

- Python
- LangGraph
- LangChain
- Google Gemini via `langchain-google-genai`
- yfinance
- Tavily
- python-dotenv

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Do not commit `.env` or real API keys.

## Run

```powershell
python main.py
```

Example session:

```text
============================================================
  Company Research Agent
  Ask me about any company. Type 'quit' to exit.
============================================================

You: Apple

  [+] Ticker resolved (AAPL)
  [+] Business model done
  [+] News & sentiment done
  [+] Financials done
  [+] Report ready

## 1. What Does This Company Do?
Apple designs, manufactures, and markets consumer electronics, software, and services...

## 5. One-Line Verdict
Apple is attractive for a long-term investor because its large installed base and services ecosystem create a durable moat.
```

If the same resolved ticker is requested again in the same session, the app returns the cached report before the expensive research agents run:

```text
You: Apple

  [+] Ticker resolved (AAPL)

[Showing cached report for AAPL]
```

## Output Format

The final report keeps a five-section format:

1. What Does This Company Do?
2. How Does It Make Money?
3. Is the Business Healthy?
4. What Are the Risks?
5. One-Line Verdict

## Verification

Syntax check:

```powershell
python -m py_compile main.py graph.py state.py agents\fundamental.py agents\business_model.py agents\news_sentiment.py agents\report_writer.py tools\yfinance_tools.py tools\tavily_tools.py
```

If local tests are present:

```powershell
python -m unittest discover -s tests -p "test*.py"
```

Manual acceptance checks:

- First-time company query runs ticker resolution, all three research agents, and report synthesis.
- Repeating the same resolved ticker in one session prints the cached report without running the research agents.
- Missing `GOOGLE_API_KEY` or `TAVILY_API_KEY` produces a readable startup error.
- Unknown ticker input ends cleanly with a readable error.

## Limitations

- yfinance can return sparse data for some HK tickers. The agent should state when metrics are unavailable.
- Tavily search quality depends on current web results and query phrasing.
- The session cache is in-memory only. It is cleared when the CLI exits.
- Cached reports can become stale within a long-running session. A refresh mode is planned but not implemented yet.
- Tavily's current LangChain integration may emit a deprecation warning; this does not affect the current CLI flow.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the phased context-engineering and harness-engineering plan.
