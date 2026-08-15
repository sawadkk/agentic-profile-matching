"""LangChain tool wrappers for the matching agent.

Wraps Milestone 1 (fs_tools) and Milestone 2 (JobMatcher) functionality as
LangChain tools, and adds three new tools: extract_requirements,
compare_candidates, and generate_interview_questions.

The three new tools call Claude. Unlike fs_tools (which never raises) and
the RAG pipeline (which raises typed exceptions), these tools must degrade
gracefully — a tool exception would break the graph run — so failures are
logged and a usable fallback is returned instead of raised.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.tools import tool

import fs_tools
from job_matcher import JobMatcher, JobMatcherError

# See resume_rag.py for why this is called here too, not just in app.py/cli.py.
load_dotenv()

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.environ.get("MODEL", "claude-sonnet-4-5")


class AgentToolError(Exception):
    """Raised only for programming errors in tool wiring, never for API failures."""


# Lightweight call log so the UI layer (app.py / cli.py) can show which
# tools a graph node invoked this turn. Nodes call tools directly (there's
# no LangGraph ToolNode in this graph), so this is the only place that
# order gets recorded.
_CALL_LOG: list[str] = []


def reset_call_log() -> None:
    """Clear the tool call log. Call once at the start of each user turn."""
    _CALL_LOG.clear()


def get_call_log() -> list[str]:
    """Return a copy of the tool call log recorded since the last reset."""
    return list(_CALL_LOG)


def _log_call(name: str) -> None:
    _CALL_LOG.append(name)


def _get_anthropic_client() -> Optional[Any]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        return anthropic.Anthropic(api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to initialize Anthropic client: %s", exc)
        return None


def _call_claude_tool(
    client: Any,
    tool_schema: dict[str, Any],
    prompt: str,
    model: str = _DEFAULT_MODEL,
    max_tokens: int = 1500,
) -> Optional[dict[str, Any]]:
    """Call Claude with a single forced tool call. Returns None on any failure."""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            tools=[tool_schema],
            tool_choice={"type": "tool", "name": tool_schema["name"]},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Claude tool call '%s' failed: %s", tool_schema["name"], exc)
        return None


# --- Milestone 1 tools (fs_tools) --------------------------------------


@tool
def read_file(filepath: str) -> dict[str, Any]:
    """Read a .txt, .pdf, or .docx file and return its text content and metadata."""
    _log_call("read_file")
    return fs_tools.read_file(filepath)


@tool
def list_files(directory: str, extension: Optional[str] = None) -> list[dict[str, Any]]:
    """List files in a directory (non-recursive), optionally filtered by extension."""
    _log_call("list_files")
    return fs_tools.list_files(directory, extension)


@tool
def write_file(filepath: str, content: str) -> dict[str, Any]:
    """Write UTF-8 text content to a file, creating parent directories as needed."""
    _log_call("write_file")
    return fs_tools.write_file(filepath, content)


@tool
def search_in_file(filepath: str, keyword: str) -> dict[str, Any]:
    """Case-insensitively search a file's content for a keyword, with context."""
    _log_call("search_in_file")
    return fs_tools.search_in_file(filepath, keyword)


# --- Milestone 2 tool (JobMatcher) --------------------------------------

_job_matcher: Optional[JobMatcher] = None


def _get_job_matcher() -> JobMatcher:
    global _job_matcher
    if _job_matcher is None:
        _job_matcher = JobMatcher()
    return _job_matcher


@tool
def rag_search(query: str, k: int = 10) -> dict[str, Any]:
    """Search the resume collection for candidates matching a job description query."""
    _log_call("rag_search")
    try:
        return _get_job_matcher().search(query, k=k)
    except JobMatcherError as exc:
        logger.warning("rag_search failed: %s", exc)
        return {"job_description": query, "top_matches": [], "error": str(exc)}


# --- New tool 1: extract_requirements -----------------------------------

_EXTRACT_REQUIREMENTS_TOOL = {
    "name": "record_requirements",
    "description": "Record structured requirements extracted from a job description.",
    "input_schema": {
        "type": "object",
        "properties": {
            "role_title": {"type": "string"},
            "must_have_skills": {"type": "array", "items": {"type": "string"}},
            "min_years": {"type": "integer"},
            "nice_to_have_skills": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["role_title", "must_have_skills", "min_years", "nice_to_have_skills"],
    },
}

_YEARS_FALLBACK_REGEX = re.compile(r"(\d{1,2})\s*\+\s*years?", re.IGNORECASE)


def _regex_fallback_requirements(jd: str) -> dict[str, Any]:
    from job_matcher import extract_critical_skills, extract_nice_to_have_skills

    years_match = _YEARS_FALLBACK_REGEX.search(jd)
    min_years = int(years_match.group(1)) if years_match else 0
    title = jd.strip().splitlines()[0].strip() if jd.strip() else "Unknown Role"

    return {
        "role_title": title,
        "must_have": {"skills": extract_critical_skills(jd), "min_years": min_years},
        "nice_to_have": {"skills": extract_nice_to_have_skills(jd)},
    }


@tool
def extract_requirements(jd: str) -> dict[str, Any]:
    """Parse a job description into must-have vs nice-to-have requirements.

    Distinguishes a "Requirements"/"Must have" block from "Nice to have"/
    "Preferred"/"Bonus". Uses Claude with a forced tool call for reliable
    structure; falls back to regex-based extraction if Claude is unavailable
    or the call fails.

    Returns:
        {"role_title", "must_have": {"skills", "min_years"}, "nice_to_have": {"skills"}}
    """
    _log_call("extract_requirements")
    client = _get_anthropic_client()
    if client is not None:
        prompt = (
            "Extract structured hiring requirements from this job description. "
            "Distinguish must-have requirements (from a Requirements/Must have/"
            "Qualifications section) from nice-to-have ones (Nice to have/"
            "Preferred/Bonus section). Extract the minimum years of experience "
            "required as an integer (0 if unspecified).\n\n"
            f"Job description:\n{jd}"
        )
        result = _call_claude_tool(client, _EXTRACT_REQUIREMENTS_TOOL, prompt)
        if result is not None:
            return {
                "role_title": result.get("role_title", "Unknown Role"),
                "must_have": {
                    "skills": list(result.get("must_have_skills", [])),
                    "min_years": int(result.get("min_years", 0)),
                },
                "nice_to_have": {"skills": list(result.get("nice_to_have_skills", []))},
            }

    logger.warning("extract_requirements falling back to regex extraction")
    return _regex_fallback_requirements(jd)


# --- New tool 2: compare_candidates --------------------------------------


def _fetch_candidate_chunks(candidate_name: str) -> list[dict[str, Any]]:
    return _get_job_matcher().get_candidate_chunks(candidate_name)


_COMPARE_CANDIDATES_TOOL = {
    "name": "record_comparison",
    "description": "Record a structured head-to-head candidate comparison.",
    "input_schema": {
        "type": "object",
        "properties": {
            "narrative": {"type": "string"},
        },
        "required": ["narrative"],
    },
}


@tool
def compare_candidates(candidate_names: list[str]) -> dict[str, Any]:
    """Produce a head-to-head comparison of 2-5 candidates.

    Pulls each candidate's chunks from ChromaDB and returns a shared
    attribute table (years, matched skills, education, key strengths) plus
    a short narrative on how they differ. Falls back to a table-only
    comparison (no narrative) if Claude is unavailable.
    """
    _log_call("compare_candidates")
    if not (2 <= len(candidate_names) <= 5):
        return {
            "candidate_names": candidate_names,
            "attribute_table": [],
            "narrative": "",
            "error": "compare_candidates supports between 2 and 5 candidates.",
        }

    attribute_table: list[dict[str, Any]] = []
    for name in candidate_names:
        chunks = _fetch_candidate_chunks(name)
        if not chunks:
            attribute_table.append(
                {
                    "candidate_name": name,
                    "experience_years": None,
                    "education": None,
                    "skills": [],
                    "error": "No indexed chunks found for this candidate.",
                }
            )
            continue

        meta = chunks[0]["metadata"]
        skills_str = meta.get("skills", "")
        skills = [s.strip() for s in skills_str.split(",") if s.strip()]
        attribute_table.append(
            {
                "candidate_name": name,
                "experience_years": meta.get("experience_years"),
                "education": meta.get("education"),
                "skills": skills,
            }
        )

    narrative = _compare_candidates_narrative(candidate_names, attribute_table)
    return {
        "candidate_names": candidate_names,
        "attribute_table": attribute_table,
        "narrative": narrative,
    }


def _compare_candidates_narrative(
    candidate_names: list[str], attribute_table: list[dict[str, Any]]
) -> str:
    client = _get_anthropic_client()
    if client is None:
        return "Claude unavailable — see attribute table for a structured comparison."

    summary_lines = []
    for row in attribute_table:
        summary_lines.append(
            f"- {row['candidate_name']}: {row.get('experience_years')} years, "
            f"education: {row.get('education')}, skills: {', '.join(row.get('skills', []))}"
        )

    prompt = (
        "Compare these candidates head-to-head based only on the attributes "
        "given below. Write a short (3-5 sentence) narrative describing how "
        "they differ. Do not invent experience not listed here.\n\n"
        + "\n".join(summary_lines)
    )
    result = _call_claude_tool(client, _COMPARE_CANDIDATES_TOOL, prompt)
    if result is None:
        return "Claude call failed — see attribute table for a structured comparison."
    return result.get("narrative", "")


# --- New tool 3: generate_interview_questions ----------------------------

_INTERVIEW_QUESTIONS_TOOL = {
    "name": "record_interview_questions",
    "description": "Record a set of targeted interview questions for a candidate.",
    "input_schema": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "probes": {"type": "string", "enum": ["strength", "gap"]},
                        "rationale": {"type": "string"},
                    },
                    "required": ["question", "probes", "rationale"],
                },
            }
        },
        "required": ["questions"],
    },
}

_FALLBACK_QUESTIONS = [
    {
        "question": "Walk me through your most recent project relevant to this role.",
        "probes": "strength",
        "rationale": "General fallback question — Claude was unavailable.",
    },
    {
        "question": "What is an area of this role's requirements you have the least experience with?",
        "probes": "gap",
        "rationale": "General fallback question — Claude was unavailable.",
    },
]


@tool
def generate_interview_questions(candidate_name: str, requirements: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Generate 5-7 targeted screening questions for a candidate.

    Mixes questions probing claimed strengths and questions probing gaps
    against the current job requirements. Grounded in the candidate's
    retrieved resume chunks. Falls back to generic questions if Claude is
    unavailable or the candidate has no indexed chunks.
    """
    _log_call("generate_interview_questions")
    chunks = _fetch_candidate_chunks(candidate_name)
    if not chunks:
        return {
            "candidate_name": candidate_name,
            "questions": _FALLBACK_QUESTIONS,
            "warning": f"No indexed chunks found for {candidate_name}; using generic questions.",
        }

    resume_text = "\n\n".join(c["text"] for c in chunks)
    requirements = requirements or {}

    client = _get_anthropic_client()
    if client is None:
        logger.warning("generate_interview_questions: no Claude client, using fallback")
        return {"candidate_name": candidate_name, "questions": _FALLBACK_QUESTIONS}

    prompt = (
        f"Given this candidate's resume content and the job requirements below, "
        f"generate 5-7 targeted screening questions. Mix questions that probe "
        f"claimed strengths with questions that probe gaps against the "
        f"requirements. Ground every question in the actual resume text or "
        f"requirements — never invent experience.\n\n"
        f"Requirements: {json.dumps(requirements)}\n\n"
        f"Resume content:\n{resume_text[:6000]}"
    )
    result = _call_claude_tool(client, _INTERVIEW_QUESTIONS_TOOL, prompt)
    if result is None:
        return {"candidate_name": candidate_name, "questions": _FALLBACK_QUESTIONS}

    return {"candidate_name": candidate_name, "questions": result.get("questions", _FALLBACK_QUESTIONS)}


ALL_TOOLS = [
    read_file,
    list_files,
    write_file,
    search_in_file,
    rag_search,
    extract_requirements,
    compare_candidates,
    generate_interview_questions,
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    with open("data/job_descriptions/senior_ml_engineer.txt", "r", encoding="utf-8") as fh:
        jd_text = fh.read()

    print("=== extract_requirements ===")
    reqs = extract_requirements.invoke({"jd": jd_text})
    print(json.dumps(reqs, indent=2))

    print("\n=== compare_candidates ===")
    comparison = compare_candidates.invoke(
        {"candidate_names": ["Elena Whitfield", "Marcus Nakamura", "Sofia Delacroix"]}
    )
    print(json.dumps(comparison, indent=2)[:1500])

    print("\n=== generate_interview_questions ===")
    questions = generate_interview_questions.invoke(
        {"candidate_name": "Elena Whitfield", "requirements": reqs}
    )
    print(json.dumps(questions, indent=2))
