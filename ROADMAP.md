# Context & Harness Engineering Roadmap

## Purpose

This roadmap upgrades the Company Research Agent from a working agentic prototype into a more reliable AI-engineering portfolio project. It focuses on two areas:

- Context engineering: improving what each agent sees, returns, trusts, and passes to the report writer.
- Harness engineering: improving orchestration, testing, observability, configuration, and operational reliability.

Use this file as a step-by-step implementation guide. Finish and verify one phase before starting the next.

## Progress

- [x] Phase 1: Reliability Baseline
  - Cache is checked after ticker resolution and before expensive research agents run.
  - Startup validates `GOOGLE_API_KEY` and `TAVILY_API_KEY`.
  - Unknown ticker routing still returns a readable error.
- [x] Phase 2: Portfolio Documentation
- [x] Phase 3: Structured Context
- [ ] Phase 4: Tests and Evaluations
- [ ] Phase 5: Source Quality and Observability
- [ ] Phase 6: Memory and Query Modes

## Priority Order

1. [Done] Fix cache placement so repeated companies avoid unnecessary agent and API calls.
2. [Done] Add a portfolio-quality `README.md` with setup, architecture, and sample output.
3. [Done] Add structured intermediate outputs from the specialist agents.
4. Add tests with mocked LLM and tool calls.
5. Add data-quality flags and source-aware synthesis.
6. Add short-term memory for follow-up questions and session context.
7. Add observability with tracing or structured run logs.
8. Add long-term memory for stable preferences, watchlists, and report history.

## Context Engineering Checklist

- Structured agent outputs: each specialist agent should return fields such as `summary`, `key_facts`, `risks`, `sources`, `data_quality`, and `confidence`, instead of only unstructured prose.
- User intent fields: carry stable intent through `AgentState`, such as `research_focus`, `time_horizon`, `report_type`, and comparison targets when supported.
- Source-aware synthesis: business and news agents should preserve source URLs, and the report writer should avoid claims that are not supported by agent evidence.
- Data-quality flags: financial and search agents should mark their data as `good`, `partial`, or `weak`, especially when yfinance returns sparse HK ticker data.
- Context compression: filter duplicate, stale, low-quality, or off-topic Tavily results before they become report-writer context.
- Short-term memory: remember the last ticker, last report, and recent user intent during a CLI session.
- Long-term memory: later persist only durable user preferences, watchlists, and report metadata. Do not store stale financial facts as durable truth.
- Stricter prompt contracts: update prompts so each agent knows its output shape and how to handle missing or weak evidence.

## Harness Engineering Checklist

- Cache placement: resolve the ticker first, check cache before expensive research nodes run, and later add a refresh path.
- Mocked tests: cover ticker resolution, unknown ticker routing, fan-out/fan-in behavior, report synthesis inputs, and tool failure behavior without real API calls.
- Typed state: replace free-text-only intermediate fields with typed structures where useful while keeping the final report as plain text.
- Per-agent error handling: allow final report generation with partial data when one specialist agent fails, and surface warnings clearly.
- Observability: add LangSmith tracing or structured logs for node start/end, latency, tool calls, failures, and selected model fallback.
- Evaluation examples: keep sample inputs and expected report traits for Apple, Tencent, an unknown company, and a ticker with sparse data.
- CLI modes: consider `--refresh`, `--no-cache`, `--mode full/news/risk/business`, and `--ticker` to skip ticker resolution.
- Config layer: centralize model names, Tavily result counts, news time window, cache behavior, and tracing toggles.
- Environment validation: fail fast with readable messages when `GOOGLE_API_KEY` or `TAVILY_API_KEY` is missing.

## Suggested Implementation Phases

### Phase 1: Reliability Baseline

Status: Done.

- [x] Fix cache placement so repeated queries avoid running the research agents.
- [x] Add startup validation for required environment variables.
- [x] Keep behavior unchanged for a normal first-time company query.

Acceptance criteria:

- [x] Asking for the same resolved ticker twice in one session uses the cached report before expensive research nodes run.
- [x] Missing environment variables produce clear startup errors.
- [x] Unknown ticker routing still ends cleanly with a readable error.

### Phase 2: Portfolio Documentation

Status: Done.

- [x] Add `README.md` with setup, environment variables, architecture, example command/session, sample report excerpt, limitations, and roadmap link.
- [x] Include a small architecture diagram or text topology.
- [x] Keep `AGENTS.md` focused on future-agent handoff and link to this roadmap if useful.

Acceptance criteria:

- [x] A reviewer can understand what the project does, how to run it, and why the architecture exists without opening the source first.
- [x] No secrets or `.env` values are included.
- [x] The README matches the current project structure.

### Phase 3: Structured Context

Status: Done.

- [x] Introduce typed structures for specialist outputs in `state.py`.
- [x] Update agent nodes and prompts so they return predictable fields.
- [x] Update the report writer to consume structured fields and preserve missing-data warnings.

Acceptance criteria:

- [x] Each specialist output includes summary, important facts, risks or caveats, source references where available, and data-quality status.
- [x] The final report uses the structured context without losing the existing five-section report format.
- [x] Missing or weak data is visible to the report writer and reflected in the final report.

### Phase 4: Tests and Evaluations

- Add tests using mocked LLM and tool responses.
- Cover graph routing, partial failures, structured-output parsing, cache behavior, and report-writer inputs.
- Add a small evaluation set of representative companies and expected report traits.

Acceptance criteria:

- Tests run without real Gemini, Tavily, or yfinance network calls.
- Core graph behavior is covered by deterministic tests.
- Evaluation examples document what a good output should contain.

### Phase 5: Source Quality and Observability

- Add source filtering or ranking before synthesis.
- Add tracing or structured run logs for graph nodes, tool calls, latency, and errors.
- Record when fallback models are used.

Acceptance criteria:

- Search-based claims can be traced back to source URLs.
- A failed or slow agent is visible in logs or traces.
- Runs are easier to debug than reading only the final report.

### Phase 6: Memory and Query Modes

- Add short-term memory for follow-up questions, last ticker, last report, and session preferences.
- Add simple CLI modes such as full report, news-only, risk-focused, or business-model-focused.
- Later add durable watchlist and user preference memory.

Acceptance criteria:

- Follow-up questions can reuse the previous ticker when the user omits it.
- Mode selection changes which agents run or how the report is written.
- Long-term memory stores stable preferences only, not stale market facts.

## Implementation Rule

Do not implement every item at once. Work in order, keep each change small, and verify the acceptance criteria for the current phase before moving to the next phase.
