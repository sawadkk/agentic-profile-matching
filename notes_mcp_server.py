"""MCP server persisting screening decisions to SQLite, across sessions.

A second, independent MCP server (Part C / "multi-MCP integration"): same
protocol, same client (mcp_client.py), completely separate process and
storage engine from filesystem_mcp_server.py. The point being demonstrated
is that the agent doesn't need to know or care — it discovers save_decision,
get_decisions, and get_candidate_history at runtime exactly the way it
discovers read_file and search_in_file, through the same tools/list call
against a different subprocess.

Tools
-----
- save_decision(candidate_name, job_role, decision, rationale) — decision
  must be one of "hire" / "no_hire" / "maybe".
- get_decisions(job_role?) — all recorded decisions, optionally filtered to
  one job role, newest first.
- get_candidate_history(candidate_name) — every decision ever recorded for
  one candidate, newest first.

Error handling follows the same convention as filesystem_mcp_server.py: a
raised exception inside a tool handler is caught by the SDK's call_tool
wrapper and converted into a successful JSON-RPC response with
`CallToolResult.isError=True` (see that module's docstring for why). An
invalid `decision` value, an empty candidate_name/job_role, or a database
error all take that path.

Transport: stdio by default (what mcp_client.py uses); `--transport http`
serves streamable HTTP for curl-based inspection, matching
filesystem_mcp_server.py's transport story for consistency.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("notes_mcp_server")

SERVER_NAME = "notes-mcp"
SERVER_VERSION = "1.0.0"
_DEFAULT_CONFIG_PATH = Path(__file__).parent / "mcp_config.json"
_VALID_DECISIONS = {"hire", "no_hire", "maybe"}


@dataclass
class Config:
    database_path: str
    base_dir: Path

    def resolved_database_path(self) -> str:
        candidate = Path(self.database_path)
        if not candidate.is_absolute():
            candidate = self.base_dir / candidate
        return str(candidate)


def _fail(message: str) -> None:
    logger.error("Configuration error: %s", message)
    sys.stderr.write(f"notes_mcp_server: configuration error: {message}\n")
    sys.exit(1)


def load_config(path: Path) -> Config:
    """Load and validate the `notes_database_path` field of mcp_config.json.

    Fails loudly on anything malformed rather than starting against a
    broken or ambiguous database path.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        _fail(f"Config file not found: {path}")
        raise AssertionError("unreachable")
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        _fail(f"Config file {path} is not valid JSON: {exc}")
        raise AssertionError("unreachable")

    if not isinstance(raw, dict) or "notes_database_path" not in raw:
        _fail(f"Config file {path} missing required field: 'notes_database_path'")
    database_path = raw["notes_database_path"]
    if not isinstance(database_path, str) or not database_path:
        _fail("Config field 'notes_database_path' must be a non-empty string")

    database_path = os.environ.get("MCP_NOTES_DATABASE_PATH", database_path)

    return Config(database_path=database_path, base_dir=path.parent.resolve())


CONFIG: Config  # set in main() before the server starts handling requests


# --- SQLite persistence -------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_name TEXT NOT NULL,
    job_role TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('hire', 'no_hire', 'maybe')),
    rationale TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_candidate ON decisions (candidate_name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_decisions_job_role ON decisions (job_role COLLATE NOCASE);
"""


def _connect(config: Config) -> sqlite3.Connection:
    db_path = config.resolved_database_path()
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "candidate_name": row["candidate_name"],
        "job_role": row["job_role"],
        "decision": row["decision"],
        "rationale": row["rationale"],
        "created_at": row["created_at"],
    }


def _tool_save_decision(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    candidate_name = (arguments.get("candidate_name") or "").strip()
    job_role = (arguments.get("job_role") or "").strip()
    decision = arguments.get("decision")
    rationale = arguments.get("rationale") or ""

    if not candidate_name:
        raise ValueError("Invalid params: candidate_name must be non-empty")
    if not job_role:
        raise ValueError("Invalid params: job_role must be non-empty")
    if decision not in _VALID_DECISIONS:
        raise ValueError(f"Invalid params: decision must be one of {sorted(_VALID_DECISIONS)}, got {decision!r}")

    created_at = datetime.now(timezone.utc).isoformat()
    conn = _connect(config)
    try:
        cursor = conn.execute(
            "INSERT INTO decisions (candidate_name, job_role, decision, rationale, created_at) VALUES (?, ?, ?, ?, ?)",
            (candidate_name, job_role, decision, rationale, created_at),
        )
        conn.commit()
        row_id = cursor.lastrowid
    finally:
        conn.close()

    return {
        "success": True,
        "id": row_id,
        "candidate_name": candidate_name,
        "job_role": job_role,
        "decision": decision,
        "rationale": rationale,
        "created_at": created_at,
    }


def _tool_get_decisions(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    job_role = arguments.get("job_role")
    conn = _connect(config)
    try:
        if job_role:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE job_role = ? COLLATE NOCASE ORDER BY created_at DESC", (job_role,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM decisions ORDER BY created_at DESC").fetchall()
    finally:
        conn.close()

    decisions = [_row_to_dict(r) for r in rows]
    return {"job_role_filter": job_role, "decisions": decisions, "count": len(decisions)}


def _tool_get_candidate_history(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    candidate_name = (arguments.get("candidate_name") or "").strip()
    if not candidate_name:
        raise ValueError("Invalid params: candidate_name must be non-empty")

    conn = _connect(config)
    try:
        rows = conn.execute(
            "SELECT * FROM decisions WHERE candidate_name = ? COLLATE NOCASE ORDER BY created_at DESC", (candidate_name,)
        ).fetchall()
    finally:
        conn.close()

    decisions = [_row_to_dict(r) for r in rows]
    return {"candidate_name": candidate_name, "previously_screened": len(decisions) > 0, "decisions": decisions, "count": len(decisions)}


# --- Tool registry --------------------------------------------------------

_TOOLS: list[types.Tool] = [
    types.Tool(
        name="save_decision",
        description="Persist a screening decision (hire/no_hire/maybe) for a candidate against a job role, with rationale.",
        inputSchema={
            "type": "object",
            "properties": {
                "candidate_name": {"type": "string"},
                "job_role": {"type": "string"},
                "decision": {"type": "string", "enum": sorted(_VALID_DECISIONS)},
                "rationale": {"type": "string"},
            },
            "required": ["candidate_name", "job_role", "decision", "rationale"],
        },
    ),
    types.Tool(
        name="get_decisions",
        description="Retrieve past screening decisions, optionally filtered to one job role, newest first.",
        inputSchema={
            "type": "object",
            "properties": {"job_role": {"type": "string"}},
            "required": [],
        },
    ),
    types.Tool(
        name="get_candidate_history",
        description="Retrieve every decision ever recorded for a candidate (across all job roles and sessions), newest first.",
        inputSchema={
            "type": "object",
            "properties": {"candidate_name": {"type": "string"}},
            "required": ["candidate_name"],
        },
    ),
]

_TOOL_DISPATCH: dict[str, Any] = {
    "save_decision": _tool_save_decision,
    "get_decisions": _tool_get_decisions,
    "get_candidate_history": _tool_get_candidate_history,
}


# --- MCP server wiring --------------------------------------------------------

server: Server = Server(SERVER_NAME, version=SERVER_VERSION)


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return _TOOLS


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    arguments = arguments or {}
    func = _TOOL_DISPATCH.get(name)
    if func is None:
        raise ValueError(f"Unknown tool: {name}")
    # sqlite3 is blocking; keep it off the event loop like the filesystem server does.
    return await asyncio.to_thread(func, arguments, CONFIG)


# --- Entry point ---------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Notes MCP server (SQLite-backed screening decisions)")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP transport only")
    parser.add_argument("--port", type=int, default=8766, help="HTTP transport only")
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG_PATH))
    return parser.parse_args(argv)


async def _run_http(host: str, port: int) -> None:
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Mount

    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    session_manager = StreamableHTTPSessionManager(app=server, json_response=True, stateless=True)

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        async with session_manager.run():
            logger.info("notes MCP server (streamable HTTP) listening on http://%s:%s/mcp/", host, port)
            yield

    starlette_app = Starlette(routes=[Mount("/mcp", app=session_manager.handle_request)], lifespan=lifespan)
    uvicorn_config = uvicorn.Config(starlette_app, host=host, port=port, log_level="warning")
    await uvicorn.Server(uvicorn_config).serve()


async def main(argv: Optional[list[str]] = None) -> None:
    global CONFIG
    args = _parse_args(argv)
    CONFIG = load_config(Path(args.config))
    # Fail fast if the database file/path is unusable, rather than on the first tool call.
    _connect(CONFIG).close()

    if args.transport == "stdio":
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(notification_options=NotificationOptions()),
            )
    else:
        await _run_http(args.host, args.port)


if __name__ == "__main__":
    asyncio.run(main())
