"""Section-aware RAG indexing pipeline for resumes.

Milestone 2. Chunks resumes by detected section headers, embeds each chunk
locally with sentence-transformers, and stores them in a persistent ChromaDB
collection with rich metadata (skills, experience_years, education, section
type) so job_matcher.py can retrieve and score candidates.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

# ChromaDB requires sqlite3 >= 3.35.0; many Linux systems (incl. this one)
# ship an older system sqlite3. pysqlite3-binary bundles a modern build —
# swap it in before chromadb imports sqlite3 internally.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import fs_tools

# Loaded here (not just in app.py/cli.py) so this module works correctly
# whether it's run directly (`python resume_rag.py`) or imported — module-
# level env reads elsewhere (MODEL) and the lazy ANTHROPIC_API_KEY read in
# MetadataExtractor both depend on .env having been loaded by this point.
load_dotenv()

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "resumes"

# Whole-line heading detection: the line, stripped, must fullmatch one of
# these (case-insensitive) and be under 60 chars. This is deliberately a
# closed vocabulary rather than a loose "looks like a heading" heuristic —
# it's what stops body text like "Experience with Kubernetes" from being
# mis-read as a section boundary.
_SECTION_PATTERNS: dict[str, str] = {
    "summary": r"(summary|objective)s?:?",
    "experience": r"(experience|work experience|employment|work history):?",
    "education": r"education:?",
    "skills": r"(skills|technical skills):?",
    "projects": r"projects?:?",
    "certifications": r"certifications?:?",
}
_SECTION_TYPE_BY_MATCH: list[tuple[re.Pattern, str]] = [
    (re.compile("^" + pattern + "$", re.IGNORECASE), section_type)
    for section_type, pattern in _SECTION_PATTERNS.items()
]

_MAX_HEADING_LEN = 60
_UNSTRUCTURED_WINDOW_WORDS = 300
_UNSTRUCTURED_OVERLAP_WORDS = 50

# Regex fallback for experience years when Claude extraction is unavailable
# or fails. Matters a lot in practice: defaulting to 0 makes every
# must-have-years filter reject the whole corpus.
_YEARS_REGEX = re.compile(
    r"(\d{1,2})\+?\s*years?\s+of\s+experience", re.IGNORECASE
)

# Small curated skill vocabulary used for regex-fallback skill extraction.
# Word-boundary aware so "Go" doesn't match inside "Django", "Angular", etc.
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

_EDUCATION_REGEX = re.compile(
    r"(B\.?S\.?|B\.?A\.?|M\.?S\.?|M\.?B\.?A\.?|Ph\.?D\.?)[^\n]{0,80}", re.IGNORECASE
)


class ResumeRAGError(Exception):
    """Raised when resume ingestion or indexing fails."""


@dataclass
class Chunk:
    """A single section-aware chunk of a resume, ready to embed."""

    text: str
    section_type: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _is_heading(line: str) -> Optional[str]:
    """Return the section_type if `line` is a whole-line section heading."""
    stripped = line.strip()
    if not stripped or len(stripped) >= _MAX_HEADING_LEN:
        return None
    for pattern, section_type in _SECTION_TYPE_BY_MATCH:
        if pattern.fullmatch(stripped):
            return section_type
    return None


def chunk_resume(text: str) -> list[Chunk]:
    """Split resume text into section-aware chunks.

    Headings are detected via whole-line regex fullmatch so body text
    mentioning a section name (e.g. "Experience with Kubernetes") is never
    mistaken for a heading. If no headings are found at all, falls back to
    fixed-size overlapping windows tagged "unstructured".
    """
    lines = text.split("\n")
    heading_indices: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        section_type = _is_heading(line)
        if section_type:
            heading_indices.append((i, section_type))

    if not heading_indices:
        return _chunk_unstructured(text)

    chunks: list[Chunk] = []
    chunk_index = 0

    first_heading_line = heading_indices[0][0]
    pre_heading_text = "\n".join(lines[:first_heading_line]).strip()
    if pre_heading_text:
        chunks.append(Chunk(text=pre_heading_text, section_type="header", chunk_index=chunk_index))
        chunk_index += 1

    for pos, (start_line, section_type) in enumerate(heading_indices):
        end_line = (
            heading_indices[pos + 1][0]
            if pos + 1 < len(heading_indices)
            else len(lines)
        )
        section_text = "\n".join(lines[start_line:end_line]).strip()
        if section_text:
            chunks.append(
                Chunk(text=section_text, section_type=section_type, chunk_index=chunk_index)
            )
            chunk_index += 1

    return chunks


def _chunk_unstructured(text: str) -> list[Chunk]:
    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    step = _UNSTRUCTURED_WINDOW_WORDS - _UNSTRUCTURED_OVERLAP_WORDS
    chunk_index = 0
    start = 0
    while start < len(words):
        window = words[start : start + _UNSTRUCTURED_WINDOW_WORDS]
        chunks.append(
            Chunk(text=" ".join(window), section_type="unstructured", chunk_index=chunk_index)
        )
        chunk_index += 1
        if start + _UNSTRUCTURED_WINDOW_WORDS >= len(words):
            break
        start += step

    return chunks


def _regex_extract_years(text: str) -> Optional[int]:
    match = _YEARS_REGEX.search(text)
    if match:
        return int(match.group(1))
    return None


def _regex_extract_skills(text: str) -> list[str]:
    found: list[str] = []
    for skill, pattern in _SKILL_REGEXES:
        if pattern.search(text):
            found.append(skill)
    return found


def _regex_extract_education(text: str) -> Optional[str]:
    match = _EDUCATION_REGEX.search(text)
    if match:
        return match.group(0).strip()
    return None


class MetadataExtractor:
    """Extracts candidate metadata (skills, years, education) from text.

    Extraction order: Claude (forced tool call) -> regex fallback -> hard
    defaults. The regex tier matters: falling straight to a hard default of
    0 years would make every must-have-years filter reject the whole corpus.
    """

    def __init__(self, client: Optional[Any] = None, model: Optional[str] = None) -> None:
        self._client = client
        self._model = model or os.environ.get("MODEL", "claude-sonnet-4-5")
        if self._client is None:
            self._client = self._build_default_client()

    def _build_default_client(self) -> Optional[Any]:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        try:
            import anthropic

            return anthropic.Anthropic(api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialize Anthropic client: %s", exc)
            return None

    def extract(self, full_text: str) -> dict[str, Any]:
        """Extract {skills, experience_years, education} from resume text."""
        result = self._extract_via_claude(full_text)
        if result is not None:
            return result

        logger.debug("Falling back to regex extraction for metadata")
        skills = _regex_extract_skills(full_text)
        years = _regex_extract_years(full_text)
        education = _regex_extract_education(full_text)

        return {
            "skills": skills,
            "experience_years": years if years is not None else 0,
            "education": education or "Unknown",
        }

    def _extract_via_claude(self, full_text: str) -> Optional[dict[str, Any]]:
        if self._client is None:
            return None

        tool = {
            "name": "record_resume_metadata",
            "description": "Record structured metadata extracted from a resume.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "skills": {"type": "array", "items": {"type": "string"}},
                    "experience_years": {"type": "integer"},
                    "education": {"type": "string"},
                },
                "required": ["skills", "experience_years", "education"],
            },
        }
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                tools=[tool],
                tool_choice={"type": "tool", "name": "record_resume_metadata"},
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Extract structured metadata from this resume. "
                            "List all technical/professional skills mentioned, "
                            "the candidate's total years of professional "
                            "experience (integer), and their highest degree "
                            "and institution as a single string.\n\n"
                            f"Resume:\n{full_text[:8000]}"
                        ),
                    }
                ],
            )
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    data = block.input
                    return {
                        "skills": list(data.get("skills", [])),
                        "experience_years": int(data.get("experience_years", 0)),
                        "education": str(data.get("education", "Unknown")),
                    }
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Claude metadata extraction failed, falling back: %s", exc)
            return None


def load_embedder(model_name: str) -> SentenceTransformer:
    """Load a SentenceTransformer, preferring the local cache.

    SentenceTransformer's default loading path makes several HTTP HEAD
    requests to the Hub on every instantiation to check for updates, even
    when the model is already cached locally — on a slow connection that's
    real wall-clock time for no benefit once the model has been downloaded
    once. Try local-only first; fall back to the network path so a
    first-time download still works.
    """
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:  # noqa: BLE001
        return SentenceTransformer(model_name)


class ResumeRAG:
    """Ingests resumes into a persistent, section-aware ChromaDB collection."""

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        metadata_extractor: Optional[MetadataExtractor] = None,
    ) -> None:
        try:
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._client.get_or_create_collection(collection_name)
            self._embedder = load_embedder(embedding_model_name)
        except Exception as exc:  # noqa: BLE001
            raise ResumeRAGError(f"Failed to initialize ResumeRAG: {exc}") from exc

        self._metadata_extractor = metadata_extractor or MetadataExtractor()

    def ingest_directory(self, directory: str) -> dict[str, Any]:
        """Ingest every supported resume file in `directory`.

        Returns:
            {"ingested": int, "failed": int, "errors": [{"filepath", "error"}]}
        """
        files = fs_tools.list_files(directory)
        ingested = 0
        failed = 0
        errors: list[dict[str, str]] = []

        for entry in files:
            filepath = entry["path"]
            if os.path.splitext(filepath)[1].lower() not in {".txt", ".pdf", ".docx"}:
                continue
            result = self.ingest_resume(filepath)
            if result["success"]:
                ingested += 1
            else:
                failed += 1
                errors.append({"filepath": filepath, "error": result["error"]})

        return {"ingested": ingested, "failed": failed, "errors": errors}

    def ingest_resume(self, filepath: str) -> dict[str, Any]:
        """Ingest a single resume: read, chunk, embed, and store.

        Idempotent: any existing chunks for this resume_path are deleted
        before the new ones are added, so re-ingestion doesn't duplicate.

        Returns:
            {"success": True, "chunk_count": int} or {"success": False, "error": str}
        """
        read_result = fs_tools.read_file(filepath)
        if not read_result["success"]:
            return {"success": False, "error": read_result["error"]}

        content = read_result["content"]
        if not content.strip():
            return {"success": False, "error": f"Empty content: {filepath}"}

        candidate_name = self._infer_candidate_name(content, filepath)
        metadata = self._metadata_extractor.extract(content)
        chunks = chunk_resume(content)
        if not chunks:
            return {"success": False, "error": f"No chunks produced for {filepath}"}

        try:
            self._delete_existing(filepath)

            ids: list[str] = []
            documents: list[str] = []
            embeddings: list[list[float]] = []
            metadatas: list[dict[str, Any]] = []

            for chunk in chunks:
                chunk_id = f"{filepath}::{chunk.chunk_index}"
                ids.append(chunk_id)
                documents.append(chunk.text)
                embeddings.append(self._embedder.encode(chunk.text).tolist())
                metadatas.append(
                    {
                        "resume_path": filepath,
                        "candidate_name": candidate_name,
                        "section_type": chunk.section_type,
                        "chunk_index": chunk.chunk_index,
                        "skills": ",".join(metadata["skills"]),
                        "experience_years": metadata["experience_years"],
                        "education": metadata["education"],
                    }
                )

            self._collection.add(
                ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
            )
        except Exception as exc:  # noqa: BLE001
            raise ResumeRAGError(f"Failed to ingest {filepath}: {exc}") from exc

        return {"success": True, "chunk_count": len(chunks)}

    def _delete_existing(self, filepath: str) -> None:
        try:
            self._collection.delete(where={"resume_path": filepath})
        except Exception as exc:  # noqa: BLE001
            logger.debug("No existing chunks to delete for %s (%s)", filepath, exc)

    @staticmethod
    def _infer_candidate_name(content: str, filepath: str) -> str:
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped:
                return stripped[:80]
        return os.path.splitext(os.path.basename(filepath))[0]

    def get_collection_stats(self) -> dict[str, Any]:
        """Return basic stats about the collection: chunk and resume counts."""
        try:
            count = self._collection.count()
            sample = self._collection.get(limit=min(count, 5000))
            resume_paths = {
                m.get("resume_path") for m in sample.get("metadatas", []) if m
            }
        except Exception as exc:  # noqa: BLE001
            raise ResumeRAGError(f"Failed to get collection stats: {exc}") from exc

        return {"chunk_count": count, "resume_count": len(resume_paths)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rag = ResumeRAG()
    result = rag.ingest_directory("data/resumes")
    print(f"Ingested: {result['ingested']}, Failed: {result['failed']}")
    if result["errors"]:
        print("Errors:")
        for err in result["errors"]:
            print(f"  {err['filepath']}: {err['error']}")
    print(rag.get_collection_stats())
