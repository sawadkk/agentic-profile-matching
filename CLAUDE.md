# agentic-profile-matching

LangGraph agent that screens resumes against job descriptions through a
multi-round, conversational workflow. Milestone 4 of a 4-part build.

## Stack

Python 3.11, LangGraph, langchain-openai + openai SDK (via OpenRouter,
OpenAI-API-compatible), ChromaDB, sentence-transformers, Streamlit, MCP
(official `mcp` SDK + `langchain-mcp-adapters`), watchdog, aiofiles.

## Code style

- Google-style docstrings on every public module, class, and function.
- Type hints everywhere (`from __future__ import annotations` at the top of modules).
- `fs_tools.py` never raises — every function returns a status dict/list.
- Pipeline and agent modules raise typed exceptions: `ResumeRAGError`
  (resume_rag.py), `JobMatcherError` (job_matcher.py), `AgentError`
  (matching_agent.py).
- Agent tools (`agent_tools.py`) degrade gracefully rather than raising —
  a tool exception would break the graph run — logging a warning and
  returning a usable fallback instead.

## Layout

Flat, single-package layout (no `src/`):

- `fs_tools.py` — Milestone 1 filesystem tools. Still used, but only from
  inside `filesystem_mcp_server.py` now, not imported by the agent.
- `llm_client.py` — shared OpenRouter client construction (`build_openai_client`,
  `get_model`) and the 429 retry-with-backoff wrapper (`call_with_retry`),
  used by resume_rag.py and agent_tools.py.
- `resume_rag.py` — Milestone 2 section-aware chunking + ChromaDB indexing.
- `job_matcher.py` — Milestone 2 hybrid semantic + keyword matching.
- `agent_tools.py` — LangChain tool wrappers + 3 new LLM-backed tools
  (extract_requirements, compare_candidates, generate_interview_questions).
- `mcp_config.json` — config for both MCP servers (allowed_directories,
  size/concurrency limits, server launch commands); env-var overridable.
- `filesystem_mcp_server.py` — Milestone 4 MCP server wrapping fs_tools.py:
  the 4 Milestone-1 tools plus `watch_directory`/`get_watch_events`
  (watchdog-based) and `batch_process` (concurrent), plus the resume corpus
  exposed as MCP resources. stdio by default; `--transport http` for curl.
- `notes_mcp_server.py` — Milestone 4 second MCP server, SQLite-backed,
  persisting screening decisions (`save_decision`, `get_decisions`,
  `get_candidate_history`) across sessions.
- `mcp_client.py` — `MCPClientManager`: launches both MCP servers, discovers
  their tools/resources at runtime (never hardcoded), and is the agent's
  only path to either.
- `matching_agent.py` — the LangGraph state machine (`AgentState`, nodes,
  `build_graph`, `run_turn`). Async throughout (MCP calls are async);
  `parse_jd` and `final_recommendation` call MCP directly, `human_feedback`
  routes a "history" intent through the notes server.
- `app.py` / `cli.py` — Streamlit and terminal front ends; both drive the
  graph exclusively through `matching_agent.run_turn` (now `async def`, run
  via `asyncio.run`), no duplicated logic.
- `generate_sample_data.py` — deterministic (seeded) sample resume/JD corpus.
- `test_scenarios.py` — Milestones 1-3 regression suite (6 end-to-end
  conversation flows, needs `OPENROUTER_API_KEY`).
- `test_mcp_scenarios.py` — Milestone 4 suite: 10 free protocol-level tests
  (no LLM calls) + 2 integration tests (1 LLM call total).
- `docs/state_machine.md` — graph diagram, state schema, node/tool tables.
- `docs/mcp_architecture.md` — MCP diagrams, tool/resource tables,
  tools-vs-resources, the security boundary, before/after comparison.

## Status

All milestones implemented. See `README.md` for setup, data generation,
ingestion, MCP servers, and how to run the UI/CLI/tests.
