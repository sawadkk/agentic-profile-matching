# agentic-profile-matching

LangGraph agent that screens resumes against job descriptions through a
multi-round, conversational workflow. Milestone 3 of a 3-part build.

## Stack

Python 3.11, LangGraph, langchain-anthropic (Claude), ChromaDB,
sentence-transformers, Streamlit.

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

- `fs_tools.py` — Milestone 1 filesystem tools.
- `resume_rag.py` — Milestone 2 section-aware chunking + ChromaDB indexing.
- `job_matcher.py` — Milestone 2 hybrid semantic + keyword matching.
- `agent_tools.py` — LangChain tool wrappers + 3 new Claude-backed tools
  (extract_requirements, compare_candidates, generate_interview_questions).
- `matching_agent.py` — the LangGraph state machine (`AgentState`, nodes,
  `build_graph`, `run_turn`).
- `app.py` / `cli.py` — Streamlit and terminal front ends; both drive the
  graph exclusively through `matching_agent.run_turn`, no duplicated logic.
- `generate_sample_data.py` — deterministic (seeded) sample resume/JD corpus.
- `test_scenarios.py` — the regression suite (6 end-to-end conversation flows).
- `docs/state_machine.md` — graph diagram, state schema, node/tool tables.

## Status

All milestones implemented. See `README.md` for setup, data generation,
ingestion, and how to run the UI/CLI/tests.
