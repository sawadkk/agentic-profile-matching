"""Hybrid semantic + keyword matching of candidates against a job description.

Milestone 2. Retrieves candidate chunks from the ResumeRAG ChromaDB
collection, scores each candidate with a blended semantic/keyword score,
applies must-have filtering (years and required skills), and returns a
ranked shortlist grounded in the retrieved chunk text.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

# Import resume_rag first: it patches sys.modules["sqlite3"] with
# pysqlite3-binary (chromadb needs sqlite3 >= 3.35.0) before anything here
# imports chromadb directly.
from resume_rag import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME, load_embedder

import chromadb

logger = logging.getLogger(__name__)

SEMANTIC_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4
OVER_FETCH_MULTIPLIER = 5

# Same curated vocabulary as resume_rag's regex fallback, used here to pull
# "critical skills" out of a job description. Word-boundary aware so "Go"
# doesn't match inside "Django", "Angular", etc.
_SKILL_VOCAB = [
    "Python", "Java", "Go", "JavaScript", "TypeScript", "React", "Angular",
    "Vue", "Node.js", "Django", "Flask", "Spring", "AWS", "Azure", "GCP",
    "Kubernetes", "Docker", "Terraform", "Jenkins", "CI/CD", "SQL",
    "PostgreSQL", "MongoDB", "Redis", "Kafka", "Spark", "Hadoop",
    "TensorFlow", "PyTorch", "scikit-learn", "NLP", "Computer Vision",
    "Machine Learning", "Deep Learning", "Data Analysis", "Tableau",
    "Power BI", "Excel", "Agile", "Scrum", "Product Management", "JIRA",
    "Figma", "A/B Testing", "Penetration Testing", "SIEM", "IAM", "OAuth",
    "Encryption", "Compliance", "GDPR", "iOS", "Android", "Swift", "Kotlin",
    "Flutter", "React Native", "Selenium", "pytest", "Test Automation",
    "Linux", "Bash", "Git", "GraphQL", "REST API", "Microservices", "ETL",
    "Airflow", "Snowflake", "BigQuery", "Ansible", "MLOps",
]
_SKILL_REGEXES = [
    (skill, re.compile(r"(?<![\w.])" + re.escape(skill) + r"(?![\w])", re.IGNORECASE))
    for skill in _SKILL_VOCAB
]

# "X+ years" / "minimum X years" / "at least X years" — strictest (highest)
# wins if a JD mentions more than one.
_YEARS_PATTERNS = [
    re.compile(r"(\d{1,2})\s*\+\s*years?", re.IGNORECASE),
    re.compile(r"minimum\s+(?:of\s+)?(\d{1,2})\s*\+?\s*years?", re.IGNORECASE),
    re.compile(r"at\s+least\s+(\d{1,2})\s*\+?\s*years?", re.IGNORECASE),
]

_REQUIREMENTS_HEADING = re.compile(
    r"^(requirements|must have|must-have|qualifications):?\s*$", re.IGNORECASE
)
_NICE_TO_HAVE_HEADING = re.compile(
    r"^(nice to have|nice-to-have|preferred|bonus):?\s*$", re.IGNORECASE
)
_ANY_HEADING = re.compile(r"^[A-Za-z][A-Za-z /-]{0,40}:?\s*$")

_CAPITALIZED_TERM_REGEX = re.compile(r"\b([A-Z][a-zA-Z0-9+.#]{1,30})\b")


class JobMatcherError(Exception):
    """Raised when matching against a job description fails."""


def _extract_min_years(text: str) -> Optional[int]:
    years: list[int] = []
    for pattern in _YEARS_PATTERNS:
        for match in pattern.finditer(text):
            years.append(int(match.group(1)))
    return max(years) if years else None


def _extract_block(text: str, start_heading: re.Pattern) -> str:
    """Return the text of a block starting at `start_heading` and ending at
    the next heading line (any heading, not just Requirements/Nice-to-have)."""
    lines = text.split("\n")
    start_idx = None
    for i, line in enumerate(lines):
        if start_heading.match(line.strip()):
            start_idx = i + 1
            break
    if start_idx is None:
        return ""

    end_idx = len(lines)
    for i in range(start_idx, len(lines)):
        stripped = lines[i].strip()
        if stripped and _ANY_HEADING.match(stripped) and (
            _REQUIREMENTS_HEADING.match(stripped) or _NICE_TO_HAVE_HEADING.match(stripped)
        ):
            end_idx = i
            break

    return "\n".join(lines[start_idx:end_idx])


def extract_critical_skills(job_description: str) -> list[str]:
    """Extract critical (must-have block) skills from a job description.

    Uses the curated vocabulary plus a capitalized-term heuristic that skips
    the first 3 lines (title/intro) and suppresses sub-words of already
    matched skills (e.g. won't double count "Java" inside "JavaScript").
    """
    requirements_block = _extract_block(job_description, _REQUIREMENTS_HEADING)
    search_text = requirements_block or job_description

    found: list[str] = []
    matched_spans: list[str] = []
    for skill, pattern in _SKILL_REGEXES:
        if pattern.search(search_text):
            found.append(skill)
            matched_spans.append(skill.lower())

    _STOPWORDS = {"The", "This", "We", "Our", "You", "Your", "Experience", "Requirements"}
    lines = search_text.split("\n")[3:]
    for line in lines:
        for match in _CAPITALIZED_TERM_REGEX.finditer(line):
            term = match.group(1)
            term_lower = term.lower()
            if term in found or term in _STOPWORDS or len(term) < 2:
                continue
            if any(term_lower in matched or matched in term_lower for matched in matched_spans):
                continue
            found.append(term)
            matched_spans.append(term_lower)

    return found


def extract_nice_to_have_skills(job_description: str) -> list[str]:
    """Extract skills mentioned in the Nice to have / Preferred block."""
    block = _extract_block(job_description, _NICE_TO_HAVE_HEADING)
    if not block:
        return []
    found: list[str] = []
    for skill, pattern in _SKILL_REGEXES:
        if pattern.search(block):
            found.append(skill)
    return found


def _parse_must_haves(job_description: str) -> dict[str, Any]:
    min_years = _extract_min_years(job_description)
    critical_skills = extract_critical_skills(job_description)
    return {"skills": critical_skills, "min_years": min_years or 0}


class JobMatcher:
    """Hybrid semantic + keyword search over the ResumeRAG ChromaDB collection."""

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        try:
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._client.get_or_create_collection(collection_name)
            self._embedder = load_embedder(embedding_model_name)
        except Exception as exc:  # noqa: BLE001
            raise JobMatcherError(f"Failed to initialize JobMatcher: {exc}") from exc

    def search(self, job_description: str, k: int = 10) -> dict[str, Any]:
        """Search and rank candidates against a job description.

        Over-fetches k*5 chunks, aggregates them by resume_path, scores each
        candidate, applies must-have filtering, and returns the top k.

        Returns:
            {"job_description", "top_matches": [...]}  (see `match` docstring
            for the shape of each entry)
        """
        return self.match(job_description, k=k)

    def match(self, job_description: str, k: int = 10) -> dict[str, Any]:
        """Match candidates against a job description.

        Returns:
            {
                "job_description": str,
                "top_matches": [
                    {
                        "candidate_name": str,
                        "resume_path": str,
                        "match_score": float,
                        "matched_skills": list[str],
                        "relevant_excerpts": list[str],
                        "reasoning": str,
                    },
                    ...
                ],
            }
        """
        try:
            query_embedding = self._embedder.encode(job_description).tolist()
            n_results = max(k * OVER_FETCH_MULTIPLIER, k)
            raw = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
            )
        except Exception as exc:  # noqa: BLE001
            raise JobMatcherError(f"Chroma query failed: {exc}") from exc

        must_have = _parse_must_haves(job_description)
        nice_to_have_skills = extract_nice_to_have_skills(job_description)

        candidates = self._aggregate_by_candidate(raw)
        scored = self._score_candidates(candidates, must_have["skills"])
        filtered = self._apply_must_have_filter(scored, must_have)

        filtered.sort(key=lambda c: c["match_score"], reverse=True)
        top_matches = filtered[:k]

        return {
            "job_description": job_description,
            "top_matches": top_matches,
            "must_have": must_have,
            "nice_to_have": {"skills": nice_to_have_skills},
        }

    @staticmethod
    def _aggregate_by_candidate(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
        candidates: dict[str, dict[str, Any]] = {}

        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            resume_path = meta.get("resume_path")
            if not resume_path:
                continue
            similarity = max(0.0, min(1.0, 1.0 - dist))

            entry = candidates.setdefault(
                resume_path,
                {
                    "resume_path": resume_path,
                    "candidate_name": meta.get("candidate_name", "Unknown"),
                    "experience_years": meta.get("experience_years", 0),
                    "education": meta.get("education", "Unknown"),
                    "skills": set(),
                    "chunks": [],
                    "max_similarity": 0.0,
                },
            )
            entry["chunks"].append({"text": doc, "similarity": similarity, "section_type": meta.get("section_type")})
            entry["max_similarity"] = max(entry["max_similarity"], similarity)
            skills_str = meta.get("skills", "")
            if skills_str:
                entry["skills"].update(s.strip() for s in skills_str.split(",") if s.strip())

        return candidates

    @staticmethod
    def _score_candidates(
        candidates: dict[str, dict[str, Any]], critical_skills: list[str]
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        critical_set = set(critical_skills)

        for entry in candidates.values():
            semantic_score = entry["max_similarity"]

            candidate_skills = entry["skills"]
            matched_skills = sorted(critical_set & candidate_skills)
            keyword_score = (
                len(matched_skills) / len(critical_set) if critical_set else 0.0
            )

            final = round((SEMANTIC_WEIGHT * semantic_score + KEYWORD_WEIGHT * keyword_score) * 100, 1)

            top_chunks = sorted(entry["chunks"], key=lambda c: c["similarity"], reverse=True)[:3]
            excerpts = [c["text"][:300] for c in top_chunks]

            reasoning = (
                f"Semantic similarity {semantic_score:.2f} (weight {SEMANTIC_WEIGHT}), "
                f"keyword match {len(matched_skills)}/{len(critical_set)} critical skills "
                f"(weight {KEYWORD_WEIGHT})."
            )

            results.append(
                {
                    "candidate_name": entry["candidate_name"],
                    "resume_path": entry["resume_path"],
                    "match_score": final,
                    "matched_skills": matched_skills,
                    "relevant_excerpts": excerpts,
                    "reasoning": reasoning,
                    "experience_years": entry["experience_years"],
                    "education": entry["education"],
                    "candidate_skills": sorted(candidate_skills),
                    "semantic_score": round(semantic_score * 100, 1),
                    "keyword_score": round(keyword_score * 100, 1),
                }
            )

        return results

    @staticmethod
    def _apply_must_have_filter(
        scored: list[dict[str, Any]], must_have: dict[str, Any]
    ) -> list[dict[str, Any]]:
        min_years = must_have.get("min_years", 0)
        required_skills = set(must_have.get("skills", []))
        half_required = len(required_skills) / 2 if required_skills else 0

        kept: list[dict[str, Any]] = []
        for candidate in scored:
            years = candidate.get("experience_years", 0) or 0
            if years < min_years:
                logger.debug(
                    "Excluding %s: %s years < required %s",
                    candidate["candidate_name"], years, min_years,
                )
                continue

            candidate_skills = set(candidate.get("candidate_skills", []))
            missing = required_skills - candidate_skills
            if required_skills and len(missing) > half_required:
                logger.debug(
                    "Excluding %s: missing %s/%s must-have skills",
                    candidate["candidate_name"], len(missing), len(required_skills),
                )
                continue

            kept.append(candidate)

        return kept

    def get_candidate_chunks(self, candidate_name: str) -> list[dict[str, Any]]:
        """Fetch all indexed chunks for a candidate by name.

        Returns:
            A list of {"text", "metadata"} dicts, or [] if the candidate has
            no indexed chunks or the lookup fails.
        """
        try:
            raw = self._collection.get(where={"candidate_name": candidate_name})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch chunks for %s: %s", candidate_name, exc)
            return []

        documents = raw.get("documents", [])
        metadatas = raw.get("metadatas", [])
        return [{"text": doc, "metadata": meta} for doc, meta in zip(documents, metadatas)]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    matcher = JobMatcher()
    with open("data/job_descriptions/senior_ml_engineer.txt", "r", encoding="utf-8") as fh:
        jd_text = fh.read()
    result = matcher.match(jd_text, k=5)
    print(f"Top matches for: {result['job_description'].splitlines()[0]}")
    for m in result["top_matches"]:
        print(f"  {m['candidate_name']:30s} score={m['match_score']:5.1f}  skills={m['matched_skills']}")
