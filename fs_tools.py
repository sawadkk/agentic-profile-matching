"""Filesystem tools for reading, listing, writing, and searching resume files.

Milestone 1. Every function here is designed to be called directly by an LLM
agent as a tool, so none of them raise: failures are reported as part of the
return value (a status dict, or an empty list) so a single bad file never
breaks an agent run.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

import pdfplumber
from docx import Document

_SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def read_file(filepath: str) -> dict[str, Any]:
    """Read a .txt, .pdf, or .docx file and return its text content.

    Args:
        filepath: Path to the file to read.

    Returns:
        On success: {"success": True, "content": str, "metadata": {...}}
        On failure: {"success": False, "error": str}
    """
    if not os.path.isfile(filepath):
        return {"success": False, "error": f"File not found: {filepath}"}

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        return {"success": False, "error": f"Unsupported file extension: {ext}"}

    try:
        if ext == ".txt":
            content = _read_txt(filepath)
        elif ext == ".pdf":
            content = _read_pdf(filepath)
        else:
            content = _read_docx(filepath)
    except Exception as exc:  # noqa: BLE001 - fs_tools must never raise
        return {"success": False, "error": f"Failed to read {filepath}: {exc}"}

    stat = os.stat(filepath)
    metadata = {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "extension": ext,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "char_count": len(content),
    }
    return {"success": True, "content": content, "metadata": metadata}


def _read_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _read_pdf(filepath: str) -> str:
    parts: list[str] = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
    return "\n".join(parts)


def _read_docx(filepath: str) -> str:
    doc = Document(filepath)
    return "\n".join(p.text for p in doc.paragraphs)


def list_files(directory: str, extension: Optional[str] = None) -> list[dict[str, Any]]:
    """List files in a directory (non-recursive), optionally filtered by extension.

    Args:
        directory: Directory to list.
        extension: Optional extension filter, e.g. ".pdf" or "pdf".

    Returns:
        A list of {"name", "path", "size_bytes", "modified_at"} dicts, sorted
        by name. Returns [] if the directory doesn't exist or can't be read.
    """
    if not os.path.isdir(directory):
        return []

    norm_ext = None
    if extension:
        norm_ext = extension if extension.startswith(".") else f".{extension}"
        norm_ext = norm_ext.lower()

    try:
        entries = sorted(os.listdir(directory))
    except OSError:
        return []

    results: list[dict[str, Any]] = []
    for name in entries:
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        if norm_ext and os.path.splitext(name)[1].lower() != norm_ext:
            continue
        try:
            stat = os.stat(path)
        except OSError:
            continue
        results.append(
            {
                "name": name,
                "path": path,
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        )
    return results


def write_file(filepath: str, content: str) -> dict[str, Any]:
    """Write UTF-8 text content to a file, creating parent directories as needed.

    Args:
        filepath: Destination path.
        content: Text content to write.

    Returns:
        {"success": True, "filepath": str, "bytes_written": int} or
        {"success": False, "error": str}
    """
    try:
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(content)
    except Exception as exc:  # noqa: BLE001 - fs_tools must never raise
        return {"success": False, "error": f"Failed to write {filepath}: {exc}"}

    return {
        "success": True,
        "filepath": filepath,
        "bytes_written": len(content.encode("utf-8")),
    }


def search_in_file(filepath: str, keyword: str) -> dict[str, Any]:
    """Case-insensitively search a file's text content for a keyword.

    Args:
        filepath: File to search (read via read_file, so .txt/.pdf/.docx).
        keyword: Term to search for.

    Returns:
        {"success": True, "matches": [{"context": str, "position": int}, ...],
         "match_count": int} or {"success": False, "error": str}
    """
    read_result = read_file(filepath)
    if not read_result["success"]:
        return {"success": False, "error": read_result["error"]}

    content = read_result["content"]
    if not keyword:
        return {"success": False, "error": "keyword must be non-empty"}

    matches: list[dict[str, Any]] = []
    for match in re.finditer(re.escape(keyword), content, flags=re.IGNORECASE):
        start = max(0, match.start() - 50)
        end = min(len(content), match.end() + 50)
        context = content[start:end].replace("\n", " ").strip()
        matches.append({"context": context, "position": match.start()})

    return {"success": True, "matches": matches, "match_count": len(matches)}
