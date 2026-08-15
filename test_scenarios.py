"""End-to-end conversation-flow tests for the matching agent.

These are the de-facto regression suite: each scenario drives the compiled
LangGraph agent through a realistic multi-turn conversation and asserts on
its behavior (not just that it ran without raising). Requires
ANTHROPIC_API_KEY to be set and the resume corpus to already be ingested
into ./chroma_db (see resume_rag.py).

Run directly: `python test_scenarios.py`
"""

from __future__ import annotations

import os
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage

from matching_agent import build_graph, run_turn
from resume_rag import ResumeRAG, ResumeRAGError

JD_DIR = "data/job_descriptions"


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str


def _new_config() -> dict[str, Any]:
    return {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}


def _ai_texts_since(messages: list[Any], since_human_count: int) -> list[str]:
    """Return AIMessage contents that appeared after the Nth HumanMessage (1-indexed)."""
    human_seen = 0
    out: list[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            human_seen += 1
            continue
        if isinstance(m, AIMessage) and human_seen >= since_human_count:
            out.append(m.content)
    return out


def _read_jd(filename: str) -> str:
    with open(os.path.join(JD_DIR, filename), "r", encoding="utf-8") as fh:
        return fh.read()


def test_initial_screening() -> TestResult:
    app = build_graph()
    config = _new_config()

    turn = run_turn(app, config, f"Screen candidates against {JD_DIR}/senior_ml_engineer.txt")
    state = turn["state"]

    requirements = state.get("requirements", {})
    must_have = requirements.get("must_have", {})
    shortlist = state.get("shortlist", [])

    checks = []
    checks.append(("role_title extracted", bool(requirements.get("role_title"))))
    checks.append(("must-have skills extracted", len(must_have.get("skills", [])) > 0))
    checks.append(("min_years extracted", must_have.get("min_years", 0) > 0))
    checks.append(("shortlist has candidates", len(shortlist) > 0))
    checks.append(("shortlist capped at 10", len(shortlist) <= 10))
    min_years = must_have.get("min_years", 0)
    checks.append((
        "all shortlisted candidates meet min-years bar",
        all(c.get("experience_years", 0) >= min_years for c in shortlist),
    ))

    passed = all(ok for _, ok in checks)
    detail = "; ".join(f"{name}={ok}" for name, ok in checks)
    return TestResult("1. Initial screening", passed, detail)


def test_mid_conversation_refinement() -> TestResult:
    app = build_graph()
    config = _new_config()

    run_turn(app, config, f"Screen candidates against {JD_DIR}/senior_ml_engineer.txt")

    turn2 = run_turn(app, config, "Actually, make Kubernetes a must-have skill.")
    state2 = turn2["state"]
    shortlist_after = state2["shortlist"]
    must_have_skills_after = state2["requirements"].get("must_have", {}).get("skills", [])

    ai_texts = _ai_texts_since(state2["messages"], since_human_count=2)
    delta_explained = any("what changed" in t.lower() for t in ai_texts)

    checks = [
        ("Kubernetes added to must-have", "Kubernetes" in must_have_skills_after),
        ("candidate pool re-searched", len(state2["candidates"]) > 0),
        ("shortlist still populated", len(shortlist_after) > 0),
        ("delta explanation produced", delta_explained),
    ]
    passed = all(ok for _, ok in checks)
    detail = "; ".join(f"{name}={ok}" for name, ok in checks)
    return TestResult("2. Mid-conversation refinement", passed, detail)


def test_head_to_head_comparison() -> TestResult:
    app = build_graph()
    config = _new_config()

    turn1 = run_turn(app, config, f"Screen candidates against {JD_DIR}/senior_ml_engineer.txt")
    shortlist = turn1["state"]["shortlist"]
    if len(shortlist) < 3:
        return TestResult("3. Head-to-head comparison", False, "shortlist has fewer than 3 candidates")

    top3_names = [c["candidate_name"] for c in shortlist[:3]]
    turn2 = run_turn(app, config, "Compare the top 3 matches side by side.")
    ai_texts = _ai_texts_since(turn2["state"]["messages"], since_human_count=2)
    combined = "\n".join(ai_texts)

    checks = [(f"mentions {name}", name in combined) for name in top3_names]
    checks.append(("routed via compare", turn2["state"].get("next_action") == "compare"))

    passed = all(ok for _, ok in checks)
    detail = "; ".join(f"{name}={ok}" for name, ok in checks)
    return TestResult("3. Head-to-head comparison", passed, detail)


def test_ranking_explanation() -> TestResult:
    app = build_graph()
    config = _new_config()

    turn1 = run_turn(app, config, f"Screen candidates against {JD_DIR}/senior_ml_engineer.txt")
    shortlist = turn1["state"]["shortlist"]
    if len(shortlist) < 2:
        return TestResult("4. Ranking explanation", False, "shortlist has fewer than 2 candidates")

    a, b = shortlist[0], shortlist[1]
    turn2 = run_turn(app, config, f"Why did {a['candidate_name']} rank higher than {b['candidate_name']}?")
    ai_texts = _ai_texts_since(turn2["state"]["messages"], since_human_count=2)
    combined = "\n".join(ai_texts)

    checks = [
        ("mentions both candidates", a["candidate_name"] in combined and b["candidate_name"] in combined),
        ("references a score", str(a["match_score"]) in combined or str(b["match_score"]) in combined),
        ("routed via explain", turn2["state"].get("next_action") == "explain"),
    ]
    passed = all(ok for _, ok in checks)
    detail = "; ".join(f"{name}={ok}" for name, ok in checks)
    return TestResult("4. Ranking explanation", passed, detail)


def test_full_multi_round_flow() -> TestResult:
    app = build_graph()
    config = _new_config()

    turn1 = run_turn(app, config, f"Screen candidates against {JD_DIR}/senior_ml_engineer.txt")
    round1_ok = turn1["state"]["round_number"] == 1 and len(turn1["state"]["shortlist"]) > 0

    turn2 = run_turn(app, config, "Do a deeper analysis on the top 10.")
    round2_ok = turn2["state"]["round_number"] == 2
    deep_texts = _ai_texts_since(turn2["state"]["messages"], since_human_count=2)
    round2_has_questions = any("interview questions" in t.lower() or "[gap]" in t.lower() or "[strength]" in t.lower() for t in deep_texts)

    turn3 = run_turn(app, config, "Give me a final hire recommendation.")
    round3_ok = turn3["state"]["round_number"] == 3
    final_texts = _ai_texts_since(turn3["state"]["messages"], since_human_count=3)
    round3_has_verdicts = any(
        ("HIRE" in t or "NO-HIRE" in t or "MAYBE" in t) for t in final_texts
    )

    checks = [
        ("round 1 produced a shortlist", round1_ok),
        ("round advances to 2", round2_ok),
        ("round 2 output has interview questions", round2_has_questions),
        ("round advances to 3", round3_ok),
        ("round 3 output has hire/no-hire/maybe verdicts", round3_has_verdicts),
    ]
    passed = all(ok for _, ok in checks)
    detail = "; ".join(f"{name}={ok}" for name, ok in checks)
    return TestResult("5. Full multi-round flow", passed, detail)


def test_natural_language_filter() -> TestResult:
    app = build_graph()
    config = _new_config()

    turn = run_turn(app, config, "Find me candidates with React and 3+ years of experience.")
    state = turn["state"]
    shortlist = state.get("shortlist", [])
    must_have = state.get("requirements", {}).get("must_have", {})

    checks = [
        ("React captured as a requirement", "React" in must_have.get("skills", [])),
        ("min_years captured as 3", must_have.get("min_years", 0) == 3),
        ("shortlist respects the filter", all(c.get("experience_years", 0) >= 3 for c in shortlist)),
    ]
    passed = all(ok for _, ok in checks)
    detail = "; ".join(f"{name}={ok}" for name, ok in checks)
    return TestResult("6. Natural language filter", passed, detail)


TESTS: list[Callable[[], TestResult]] = [
    test_initial_screening,
    test_mid_conversation_refinement,
    test_head_to_head_comparison,
    test_ranking_explanation,
    test_full_multi_round_flow,
    test_natural_language_filter,
]


def _ensure_corpus_ingested() -> None:
    rag = ResumeRAG()
    stats = rag.get_collection_stats()
    if stats["chunk_count"] == 0:
        print("chroma_db is empty — ingesting data/resumes/ before running tests...")
        rag.ingest_directory("data/resumes")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set — test_scenarios.py needs it to run the agent.")
        sys.exit(1)

    try:
        _ensure_corpus_ingested()
    except ResumeRAGError as exc:
        print(f"Failed to verify/ingest the resume corpus: {exc}")
        sys.exit(1)

    results: list[TestResult] = []
    for test_fn in TESTS:
        try:
            results.append(test_fn())
        except Exception as exc:  # noqa: BLE001
            results.append(TestResult(test_fn.__name__, False, f"raised {type(exc).__name__}: {exc}"))

    print("\n=== Test Results ===")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name}")
        print(f"       {r.detail}")

    passed_count = sum(1 for r in results if r.passed)
    print(f"\n{passed_count}/{len(results)} scenarios passed.")
    sys.exit(0 if passed_count == len(results) else 1)


if __name__ == "__main__":
    main()
