"""LangGraph agent for multi-round, conversational resume screening.

Wires together the RAG / JobMatcher pipeline (Milestone 2), the agent tools
(agent_tools.py), and the MCP client (mcp_client.py, Milestone 4) into a
state machine with three screening rounds: initial screen, deep analysis,
and final recommendation. Round transitions and requirement/ranking
refinements are driven by a human_feedback classifier node rather than
automatic progression or keyword matching, so the agent stays inspectable
and predictable.

This module no longer imports fs_tools directly (Milestone 4): the only
path to the filesystem is through mcp_client.MCPClientManager, which talks
to filesystem_mcp_server.py over MCP. fs_tools.py still exists and still
does the actual file I/O — it's just called from inside that server process
now, not from here. Round 3 also persists its verdicts through a second MCP
server (notes_mcp_server.py) via the same client.

Because MCP calls are async, every node that touches either MCP server
(parse_jd, final_recommendation, human_feedback) is an async def, and the
graph is driven end-to-end via LangGraph's async API (astream/aget_state/
aupdate_state) in run_turn — mixing sync and async nodes in one graph is
supported, but only if the whole graph is invoked asynchronously.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Annotated, Any, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import mcp_client
from agent_tools import (
    compare_candidates,
    extract_requirements,
    generate_interview_questions,
    get_call_log,
    log_call,
    rag_search,
    reset_call_log,
)
from llm_client import OPENROUTER_BASE_URL, get_model

# See resume_rag.py for why this is called here too, not just in app.py/cli.py.
load_dotenv()

logger = logging.getLogger(__name__)

_VALID_ROUTING_ACTIONS = {
    "refine", "rerank", "deep_screen", "final", "compare", "explain", "history", "end",
}
_VERDICT_TO_DECISION = {"hire": "hire", "no-hire": "no_hire", "maybe": "maybe"}
BORDERLINE_MARGIN = 10.0
ROUND_1_SHORTLIST_SIZE = 10


class AgentError(Exception):
    """Raised when a graph node hits an unrecoverable error."""


class AgentState(TypedDict):
    """Shared state threaded through every node in the graph.

    Attributes:
        messages: Full conversation history (LangGraph-managed via add_messages).
        job_description: The raw job description text currently in scope.
        requirements: {"role_title", "must_have": {"skills", "min_years"},
            "nice_to_have": {"skills"}}.
        candidates: Full retrieved candidate pool with scores (pre-filter).
        shortlist: Current ranked shortlist (post-filter, top N).
        round_number: 1 = initial screen, 2 = deep analysis, 3 = final recommendation.
        reports: candidate_name -> report text (round 1/2/3 outputs).
        user_feedback: Latest user refinement instruction.
        next_action: Routing hint set by human_feedback ("refine" | "rerank" |
            "deep_screen" | "final" | "compare" | "explain" | "history" | "end").
    """

    messages: Annotated[list, add_messages]
    job_description: str
    requirements: dict[str, Any]
    candidates: list[dict[str, Any]]
    shortlist: list[dict[str, Any]]
    round_number: int
    reports: dict[str, str]
    user_feedback: str
    next_action: str


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=get_model(),
        temperature=0,
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        max_retries=3,
    )


async def _get_mcp_manager() -> Optional[mcp_client.MCPClientManager]:
    """Get the MCP client manager, or None if a server is unavailable.

    mcp_client.MCPClientManager.initialize() raises MCPClientError with a
    clear message when a configured server can't be launched (bad path,
    interpreter missing, non-zero exit, etc.). Every node that touches MCP
    goes through this rather than calling mcp_client.get_manager()
    directly, so that failure degrades to a logged warning and a None the
    caller can fall back on -- matching agent_tools.py's convention that a
    tool failure must never crash a graph run. Without this, the
    exception propagates out of the node, through LangGraph's runner, and
    crashes the whole turn with a raw traceback instead of a usable
    response.
    """
    try:
        return await mcp_client.get_manager()
    except mcp_client.MCPClientError as exc:
        logger.warning("MCP unavailable: %s", exc)
        return None


def _latest_human_text(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content
        if isinstance(message, dict) and message.get("role") == "user":
            return message.get("content", "")
    return ""


# --- Nodes ----------------------------------------------------------------


_PATH_TOKEN_REGEX = re.compile(r"[^\s]+\.(?:txt|pdf|docx)\b")


def _find_referenced_filepath(text: str) -> Optional[str]:
    """Find a token in `text` that looks like a path to an existing resume/JD file.

    Handles both a bare path ("data/job_descriptions/x.txt") and a path
    embedded in a sentence ("Screen candidates against data/.../x.txt").
    This is local string disambiguation (does this token look like a real
    path at all?), not a filesystem read, so it stays a plain os.path.isfile
    check rather than an MCP round trip; the actual file content only ever
    comes back through the MCP client, below.
    """
    for match in _PATH_TOKEN_REGEX.finditer(text):
        candidate = match.group(0).rstrip(".,;:")
        if os.path.isfile(candidate):
            return candidate
    return None


async def parse_jd(state: AgentState) -> dict[str, Any]:
    """Resolve the job description: either raw text or a filepath in the latest message."""
    text = _latest_human_text(state)

    filepath = _find_referenced_filepath(text)
    if filepath:
        manager = await _get_mcp_manager()
        if manager is None:
            logger.warning("parse_jd: MCP unavailable, falling back to raw message text instead of reading %s", filepath)
        else:
            result = await manager.call_tool("read_file", filepath=filepath)
            log_call(manager.trace_label("read_file"))
            if result.get("success"):
                return {"job_description": result["content"]}
            logger.warning("parse_jd: failed to read %s via MCP: %s", filepath, result.get("error"))

    # Strip a leading instruction like "Find me candidates for this job
    # description:" if present, keeping everything after the colon.
    if ":" in text and len(text.split(":", 1)[0]) < 80:
        job_description = text.split(":", 1)[1].strip() or text
    else:
        job_description = text

    return {"job_description": job_description}


def extract_requirements_node(state: AgentState) -> dict[str, Any]:
    """Call the extract_requirements tool and store structured requirements.

    On the first pass, extracts straight from the job description. On a
    "refine" turn (previous requirements + a new user instruction both
    present in state), folds the instruction into the prompt as an update
    against the current requirements, so "make Kubernetes a must-have"
    modifies the existing list instead of re-deriving from the original JD
    and losing the change.
    """
    jd = state["job_description"]
    user_feedback = state.get("user_feedback", "")
    previous_requirements = state.get("requirements", {})

    if user_feedback and previous_requirements:
        prompt_text = (
            f"{jd}\n\n---\n"
            f"Current extracted requirements: {json.dumps(previous_requirements)}\n"
            f"User refinement instruction: {user_feedback}\n"
            f"Apply this instruction as an update to the current requirements "
            f"above (e.g. add/remove a must-have or nice-to-have skill, or "
            f"change the minimum years) and return the resulting FINAL "
            f"requirements."
        )
    else:
        prompt_text = jd

    result = extract_requirements.invoke({"jd": prompt_text})
    return {"requirements": result}


def search_resumes(state: AgentState) -> dict[str, Any]:
    """Search the resume collection via the rag_search tool, populating the candidate pool."""
    result = rag_search.invoke({"query": state["job_description"], "k": ROUND_1_SHORTLIST_SIZE})
    return {"candidates": result.get("top_matches", [])}


def _apply_requirement_filters(
    candidates: list[dict[str, Any]], requirements: dict[str, Any]
) -> list[dict[str, Any]]:
    must_have = requirements.get("must_have", {})
    min_years = must_have.get("min_years", 0) or 0
    required_skills = set(must_have.get("skills", []))
    half_required = len(required_skills) / 2 if required_skills else 0

    kept = []
    for candidate in candidates:
        years = candidate.get("experience_years", 0) or 0
        if years < min_years:
            continue
        candidate_skills = set(candidate.get("candidate_skills", []))
        missing = required_skills - candidate_skills
        if required_skills and len(missing) > half_required:
            continue
        kept.append(candidate)
    return kept


def rank_candidates(state: AgentState) -> dict[str, Any]:
    """Apply must-have filtering to the candidate pool and rank the top N.

    When this re-ranks an existing shortlist (a "refine" or "rerank" turn,
    not the first pass), also emits an AIMessage narrating what changed —
    who moved up, who moved down, who dropped off, and why. That narration
    is a hard requirement of the refinement flow, not cosmetic, so it's
    generated here rather than left to the UI layer.
    """
    candidates = state.get("candidates", [])
    requirements = state.get("requirements", {})
    previous_shortlist = state.get("shortlist", [])

    filtered = _apply_requirement_filters(candidates, requirements)
    filtered.sort(key=lambda c: c["match_score"], reverse=True)
    shortlist = filtered[:ROUND_1_SHORTLIST_SIZE]

    update: dict[str, Any] = {"shortlist": shortlist}
    if previous_shortlist:
        narrative = _rank_delta_narrative(previous_shortlist, shortlist)
        update["messages"] = [AIMessage(content=narrative)]
    return update


def _rank_delta_narrative(
    previous: list[dict[str, Any]], new: list[dict[str, Any]]
) -> str:
    """Compare a previous shortlist to a new one and narrate the delta."""
    prev_rank = {c["candidate_name"]: i for i, c in enumerate(previous)}
    new_rank = {c["candidate_name"]: i for i, c in enumerate(new)}
    prev_by_name = {c["candidate_name"]: c for c in previous}
    new_by_name = {c["candidate_name"]: c for c in new}

    moved_up: list[str] = []
    moved_down: list[str] = []
    added: list[str] = []
    for name, new_i in new_rank.items():
        if name not in prev_rank:
            added.append(
                f"{name} newly qualifies (score {new_by_name[name]['match_score']})"
            )
            continue
        prev_i = prev_rank[name]
        if new_i == prev_i:
            continue
        entry = (
            f"{name}: #{prev_i + 1} -> #{new_i + 1} "
            f"(score {prev_by_name[name]['match_score']} -> {new_by_name[name]['match_score']})"
        )
        (moved_up if new_i < prev_i else moved_down).append(entry)

    dropped = [
        f"{name} (no longer meets updated requirements)"
        for name in prev_rank
        if name not in new_rank
    ]

    lines = ["## What changed in the ranking"]
    if moved_up:
        lines.append("**Moved up:** " + "; ".join(moved_up))
    if moved_down:
        lines.append("**Moved down:** " + "; ".join(moved_down))
    if dropped:
        lines.append("**Dropped from shortlist:** " + "; ".join(dropped))
    if added:
        lines.append("**Newly added:** " + "; ".join(added))
    if len(lines) == 1:
        lines.append("No change in ranking or shortlist membership.")
    return "\n".join(lines)


def _score_breakdown_line(candidate: dict[str, Any]) -> str:
    return (
        f"semantic={candidate.get('semantic_score', 'n/a')} "
        f"keyword={candidate.get('keyword_score', 'n/a')} "
        f"final={candidate.get('match_score', 'n/a')}"
    )


def generate_report(state: AgentState) -> dict[str, Any]:
    """Generate a round-1 match report for the current shortlist.

    Per candidate: score, matched skills, gaps, strengths. Borderline
    candidates (within BORDERLINE_MARGIN of the lowest kept score) get
    improvement suggestions.
    """
    shortlist = state.get("shortlist", [])
    requirements = state.get("requirements", {})
    required_skills = set(requirements.get("must_have", {}).get("skills", []))

    cutoff = min((c["match_score"] for c in shortlist), default=0)
    reports = dict(state.get("reports", {}))

    lines = [f"## Round 1: Initial Screen — {len(shortlist)} candidates\n"]
    for candidate in shortlist:
        name = candidate["candidate_name"]
        matched = set(candidate.get("matched_skills", []))
        gaps = sorted(required_skills - matched)
        strengths = sorted(matched)

        section = [
            f"### {name} — score {candidate['match_score']}",
            f"- Score breakdown: {_score_breakdown_line(candidate)}",
            f"- Strengths: {', '.join(strengths) if strengths else 'none matched'}",
            f"- Gaps: {', '.join(gaps) if gaps else 'none'}",
        ]
        if candidate["match_score"] <= cutoff + BORDERLINE_MARGIN:
            section.append(
                f"- Borderline: closing gaps in {', '.join(gaps) if gaps else 'keyword coverage'} "
                f"would improve ranking."
            )
        report_text = "\n".join(section)
        reports[name] = report_text
        lines.append(report_text)

    return {"reports": reports, "messages": [AIMessage(content="\n\n".join(lines))]}


def deep_screen(state: AgentState) -> dict[str, Any]:
    """Round 2: deep analysis of the shortlist — strengths/gaps + interview questions."""
    shortlist = state.get("shortlist", [])
    requirements = state.get("requirements", {})
    reports = dict(state.get("reports", {}))

    lines = [f"## Round 2: Deep Analysis — {len(shortlist)} candidates\n"]
    for candidate in shortlist:
        name = candidate["candidate_name"]
        questions_result = generate_interview_questions.invoke(
            {"candidate_name": name, "requirements": requirements}
        )
        questions = questions_result.get("questions", [])

        section = [f"### {name}"]
        section.append(f"- Excerpts: {'; '.join(candidate.get('relevant_excerpts', [])[:2])}")
        section.append("- Interview questions:")
        for q in questions:
            section.append(f"  - [{q.get('probes', '?')}] {q.get('question', '')}")

        report_text = "\n".join(section)
        reports[name] = reports.get(name, "") + "\n\n" + report_text
        lines.append(report_text)

    return {"reports": reports, "round_number": 2, "messages": [AIMessage(content="\n\n".join(lines))]}


async def final_recommendation(state: AgentState) -> dict[str, Any]:
    """Round 3: hire / no-hire / maybe recommendation per candidate, with justification.

    Also persists each verdict to the notes MCP server (save_decision) so
    it's queryable in future sessions via the "history" route below. A
    persistence failure (notes server unavailable, etc.) is logged and
    surfaced in the report as a warning, not raised — the round 3 report
    itself must still complete.
    """
    shortlist = state.get("shortlist", [])
    reports = dict(state.get("reports", {}))
    job_role = state.get("requirements", {}).get("role_title") or "Unknown Role"

    cutoff = min((c["match_score"] for c in shortlist), default=0)
    lines = [f"## Round 3: Final Recommendation — {len(shortlist)} candidates\n"]

    manager = await _get_mcp_manager()
    persisted = 0
    persist_errors: list[str] = []

    for candidate in shortlist:
        name = candidate["candidate_name"]
        score = candidate["match_score"]
        if score >= cutoff + BORDERLINE_MARGIN * 2:
            verdict = "hire"
        elif score <= cutoff + BORDERLINE_MARGIN:
            verdict = "maybe"
        else:
            verdict = "hire"
        if not candidate.get("matched_skills"):
            verdict = "no-hire"

        justification = (
            f"Score {score} ({_score_breakdown_line(candidate)}); "
            f"matched skills: {', '.join(candidate.get('matched_skills', [])) or 'none'}."
        )
        section = f"### {name} — {verdict.upper()}\n- {justification}"
        reports[name] = reports.get(name, "") + "\n\n" + section
        lines.append(section)

        if manager is None:
            persist_errors.append(f"{name}: MCP notes server unavailable")
        else:
            save_result = await manager.call_tool(
                "save_decision",
                candidate_name=name,
                job_role=job_role,
                decision=_VERDICT_TO_DECISION[verdict],
                rationale=justification,
            )
            log_call(manager.trace_label("save_decision"))
            if "error" in save_result:
                persist_errors.append(f"{name}: {save_result['error']}")
            else:
                persisted += 1

    if persisted:
        lines.append(f"\n_Persisted {persisted}/{len(shortlist)} decision(s) to the notes server._")
    if persist_errors:
        logger.warning("final_recommendation: failed to persist %d decision(s): %s", len(persist_errors), persist_errors)
        lines.append(f"\n_Warning: {len(persist_errors)} decision(s) could not be persisted (notes server issue)._")

    return {"reports": reports, "round_number": 3, "messages": [AIMessage(content="\n\n".join(lines))]}


_FEEDBACK_ROUTING_TOOL = {
    "type": "function",
    "function": {
        "name": "route_feedback",
        "description": "Classify the user's latest message into a routing action.",
        "parameters": {
            "type": "object",
            "properties": {
                "next_action": {
                    "type": "string",
                    "enum": sorted(_VALID_ROUTING_ACTIONS),
                },
                "user_feedback_summary": {"type": "string"},
            },
            "required": ["next_action", "user_feedback_summary"],
        },
    },
}

_ROUTING_SYSTEM_PROMPT = """You are the routing brain for a resume-screening agent. \
Classify the user's latest message into exactly one action:

- "refine": the user changed a REQUIREMENT (added/removed a must-have or \
nice-to-have skill, changed the years threshold, changed the role). This \
requires re-extracting requirements and re-searching.
- "rerank": the user wants different ranking/weighting or filtering of the \
SAME requirements (e.g. "show me only candidates with React"), without \
changing the underlying job requirements.
- "deep_screen": the user wants round 2 deep analysis / interview questions \
for the shortlist.
- "final": the user wants round 3 hire/no-hire/maybe recommendations.
- "compare": the user wants a head-to-head comparison of two or more \
candidates (e.g. "compare the top 3", "compare Elena and Marcus").
- "explain": the user is asking why one candidate ranked above another.
- "history": the user is asking about past screening decisions for a \
candidate (e.g. "have we screened Elena before?", "what did we decide \
about Marcus previously?"), independent of the current shortlist.
- "end": none of the above (e.g. a plain acknowledgement or goodbye)."""

# "top N" in a comparison/explanation request, e.g. "compare the top 3".
_TOP_N_REGEX = re.compile(r"top\s+(\d+)", re.IGNORECASE)
# A plain "Firstname Lastname"-shaped fallback for extracting a candidate
# name from a history query when it isn't in the current shortlist.
_NAME_REGEX = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")


def _resolve_candidate_names(
    text: str, shortlist: list[dict[str, Any]], default_n: int = 3, max_n: int = 5
) -> list[str]:
    """Resolve which shortlisted candidates a comparison/explanation refers to.

    Prefers candidates named explicitly in the text; falls back to a "top N"
    mention; falls back to the top `default_n` of the current shortlist.
    """
    mentioned = [c["candidate_name"] for c in shortlist if c["candidate_name"].lower() in text.lower()]
    if len(mentioned) >= 2:
        return mentioned[:max_n]

    match = _TOP_N_REGEX.search(text)
    n = int(match.group(1)) if match else default_n
    n = max(2, min(n, max_n, len(shortlist)))
    return [c["candidate_name"] for c in shortlist[:n]]


def _handle_compare(state: AgentState, text: str) -> str:
    shortlist = state.get("shortlist", [])
    if len(shortlist) < 2:
        return "There aren't at least 2 candidates in the current shortlist to compare."

    names = _resolve_candidate_names(text, shortlist)
    result = compare_candidates.invoke({"candidate_names": names})

    lines = [f"## Comparison: {', '.join(names)}", ""]
    for row in result.get("attribute_table", []):
        lines.append(
            f"- **{row['candidate_name']}**: {row.get('experience_years')} yrs, "
            f"education: {row.get('education')}, skills: {', '.join(row.get('skills', []))}"
        )
    narrative = result.get("narrative", "")
    if narrative:
        lines += ["", narrative]
    return "\n".join(lines)


def _handle_explain(state: AgentState, text: str) -> str:
    shortlist = state.get("shortlist", [])
    if len(shortlist) < 2:
        return "There aren't at least 2 ranked candidates yet to compare."

    names = _resolve_candidate_names(text, shortlist, default_n=2, max_n=2)
    by_name = {c["candidate_name"]: c for c in shortlist}
    a, b = by_name.get(names[0]), by_name.get(names[1])
    if a is None or b is None:
        return f"I don't have both {names[0]} and {names[1]} in the current shortlist to compare."

    higher, lower = (a, b) if shortlist.index(a) < shortlist.index(b) else (b, a)
    higher_skills = ", ".join(higher.get("matched_skills", [])) or "none"
    lower_skills = ", ".join(lower.get("matched_skills", [])) or "none"
    return (
        f"**{higher['candidate_name']}** (score {higher['match_score']}) ranked above "
        f"**{lower['candidate_name']}** (score {lower['match_score']}): "
        f"semantic similarity {higher.get('semantic_score')} vs {lower.get('semantic_score')}, "
        f"keyword score {higher.get('keyword_score')} vs {lower.get('keyword_score')}. "
        f"Matched skills — {higher['candidate_name']}: {higher_skills}; "
        f"{lower['candidate_name']}: {lower_skills}."
    )


def _extract_candidate_name(text: str, shortlist: list[dict[str, Any]]) -> Optional[str]:
    """Resolve a single candidate name for a history query.

    Prefers a shortlist member named in the text (correct capitalization
    guaranteed); falls back to a bare "Firstname Lastname" pattern since a
    history query can legitimately name someone outside the current
    shortlist — that's the point of a cross-session record.
    """
    for candidate in shortlist:
        if candidate["candidate_name"].lower() in text.lower():
            return candidate["candidate_name"]
    match = _NAME_REGEX.search(text)
    return match.group(0) if match else None


async def _handle_history(state: AgentState, text: str) -> str:
    name = _extract_candidate_name(text, state.get("shortlist", []))
    if not name:
        return "I couldn't identify a candidate name in that message to look up screening history for."

    manager = await _get_mcp_manager()
    if manager is None:
        return f"Screening history for {name} is unavailable right now (notes MCP server unreachable)."

    result = await manager.call_tool("get_candidate_history", candidate_name=name)
    log_call(manager.trace_label("get_candidate_history"))

    if "error" in result:
        return f"Couldn't retrieve screening history for {name}: {result['error']}"

    decisions = result.get("decisions", [])
    if not decisions:
        return f"No prior screening history found for {name}."

    lines = [f"## Screening history for {name}", ""]
    for decision in decisions:
        date = decision.get("created_at", "")[:10]
        lines.append(
            f"- {date} — **{decision.get('decision', '?').upper()}** for "
            f"{decision.get('job_role', '?')}: {decision.get('rationale', '')}"
        )
    return "\n".join(lines)


async def human_feedback(state: AgentState) -> dict[str, Any]:
    """Interpret the user's latest message and set next_action for routing.

    For "compare", "explain", and "history" intents, also answers inline
    (grounded in the current shortlist, or — for "history" — the notes MCP
    server) since none of the three has a dedicated downstream node — all
    route back to this node or to END, so the answer has to be produced
    here.
    """
    text = _latest_human_text(state)
    llm = _get_llm()

    try:
        response = llm.bind_tools(
            [_FEEDBACK_ROUTING_TOOL], tool_choice="route_feedback"
        ).invoke(
            [
                {"role": "system", "content": _ROUTING_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ]
        )
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            logger.warning("human_feedback: model returned no tool call, defaulting to 'end'")
            next_action, feedback_summary = "end", text
        else:
            args = tool_calls[0].get("args") or {}
            next_action = args.get("next_action")
            if next_action not in _VALID_ROUTING_ACTIONS:
                logger.warning(
                    "human_feedback: invalid/missing next_action %r, defaulting to 'end'",
                    next_action,
                )
                next_action = "end"
            feedback_summary = args.get("user_feedback_summary") or text
    except Exception as exc:  # noqa: BLE001
        logger.warning("human_feedback routing failed, defaulting to 'end': %s", exc)
        next_action, feedback_summary = "end", text

    update: dict[str, Any] = {"next_action": next_action, "user_feedback": feedback_summary}
    if next_action == "compare":
        update["messages"] = [AIMessage(content=_handle_compare(state, text))]
    elif next_action == "explain":
        update["messages"] = [AIMessage(content=_handle_explain(state, text))]
    elif next_action == "history":
        update["messages"] = [AIMessage(content=await _handle_history(state, text))]
    return update


def route_after_feedback(state: AgentState) -> str:
    """Conditional edge function reading next_action out of state."""
    return state.get("next_action", "end")


# --- Graph assembly ---------------------------------------------------


def build_graph() -> Any:
    """Build and compile the LangGraph state machine with a MemorySaver checkpointer."""
    graph = StateGraph(AgentState)

    graph.add_node("parse_jd", parse_jd)
    graph.add_node("extract_requirements", extract_requirements_node)
    graph.add_node("search_resumes", search_resumes)
    graph.add_node("rank_candidates", rank_candidates)
    graph.add_node("generate_report", generate_report)
    graph.add_node("human_feedback", human_feedback)
    graph.add_node("deep_screen", deep_screen)
    graph.add_node("final_recommendation", final_recommendation)

    graph.add_edge(START, "parse_jd")
    graph.add_edge("parse_jd", "extract_requirements")
    graph.add_edge("extract_requirements", "search_resumes")
    graph.add_edge("search_resumes", "rank_candidates")
    graph.add_edge("rank_candidates", "generate_report")
    graph.add_edge("generate_report", "human_feedback")

    graph.add_conditional_edges(
        "human_feedback",
        route_after_feedback,
        {
            "refine": "extract_requirements",
            "rerank": "rank_candidates",
            "deep_screen": "deep_screen",
            "final": "final_recommendation",
            # compare/explain/history answer inline in human_feedback and
            # loop back to it — interrupt_before re-triggers, pausing for
            # the next user message instead of ending the conversation.
            "compare": "human_feedback",
            "explain": "human_feedback",
            "history": "human_feedback",
            "end": END,
        },
    )
    graph.add_edge("deep_screen", "generate_report")
    graph.add_edge("final_recommendation", END)

    # The graph pauses right before human_feedback so the caller can inject
    # the user's next message into state before the routing node reads it.
    # Without this, a fresh invoke() would run start-to-end in one shot and
    # human_feedback would have nothing new to classify.
    return graph.compile(checkpointer=MemorySaver(), interrupt_before=["human_feedback"])


def empty_state() -> AgentState:
    """A blank AgentState, used to seed a new conversation thread."""
    return {
        "messages": [],
        "job_description": "",
        "requirements": {},
        "candidates": [],
        "shortlist": [],
        "round_number": 1,
        "reports": {},
        "user_feedback": "",
        "next_action": "",
    }


async def run_turn(app: Any, config: dict[str, Any], user_input: str) -> dict[str, Any]:
    """Advance the conversation by one user turn, shared by app.py and cli.py.

    On a fresh thread, starts the graph from START with `user_input` as the
    first message. On a thread paused before human_feedback (i.e. round 1
    already ran and the agent is waiting for direction), injects
    `user_input` as the newest message and resumes from the interrupt.

    Async because parse_jd, final_recommendation, and human_feedback call
    the MCP client; the whole graph is driven via the async API
    (astream/aget_state/aupdate_state) accordingly. Callers (app.py, cli.py)
    run this via asyncio.run(...) or an existing event loop.

    Returns:
        {"state": AgentState, "trace": [{"node": str, "tools_called": [str]}]}
        Each trace entry's tools_called may include MCP-routed calls
        labeled "mcp:<server>/<tool>" (e.g. "mcp:filesystem/read_file",
        "mcp:notes/save_decision") alongside plain agent tool names, so the
        protocol boundary is visible in the trace.
    """
    reset_call_log()
    snapshot = await app.aget_state(config)
    paused_before_feedback = snapshot.next == ("human_feedback",)

    if paused_before_feedback:
        await app.aupdate_state(config, {"messages": [HumanMessage(content=user_input)]})
        stream_input: Optional[dict[str, Any]] = None
    else:
        stream_input = {**empty_state(), "messages": [HumanMessage(content=user_input)]}

    trace: list[dict[str, Any]] = []
    prev_log_len = 0
    async for step in app.astream(stream_input, config=config, stream_mode="updates"):
        for node_name in step:
            # LangGraph yields pseudo-keys like "__interrupt__" alongside
            # real node names when the graph pauses — not one of our nodes,
            # so it doesn't belong in a user-facing trace.
            if node_name.startswith("__"):
                continue
            log = get_call_log()
            trace.append({"node": node_name, "tools_called": log[prev_log_len:]})
            prev_log_len = len(log)

    final_snapshot = await app.aget_state(config)
    return {"state": final_snapshot.values, "trace": trace}


def save_graph_diagram(output_path: str = "docs/state_machine.png") -> None:
    """Render the compiled graph to a PNG via LangGraph's built-in Mermaid renderer."""
    app = build_graph()
    png_bytes = app.get_graph().draw_mermaid_png()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(png_bytes)
    print(f"Saved graph diagram to {output_path}")


async def _smoke_test() -> None:
    app = build_graph()
    config = {"configurable": {"thread_id": "smoke-test"}}

    with open("data/job_descriptions/senior_ml_engineer.txt", "r", encoding="utf-8") as fh:
        jd_text = fh.read()

    turn = await run_turn(app, config, f"Find me candidates for this job description: {jd_text}")
    state = turn["state"]

    print(f"round_number={state['round_number']}")
    print(f"requirements={json.dumps(state['requirements'], indent=2)}")
    print(f"shortlist size={len(state['shortlist'])}")
    for c in state["shortlist"][:5]:
        print(f"  {c['candidate_name']:30s} score={c['match_score']}")
    print("trace:")
    for step in turn["trace"]:
        print(f"  {step['node']:25s} tools={step['tools_called']}")


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(_smoke_test())
