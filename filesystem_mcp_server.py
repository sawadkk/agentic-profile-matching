"""MCP server exposing filesystem capabilities over JSON-RPC 2.0.

Wraps fs_tools.py (Milestone 1) behind the Model Context Protocol so agents
talk to a server process instead of importing Python functions directly.
Supports stdio (the standard local-server transport, used by the agent via
mcp_client.py) and streamable HTTP (`--transport http`, for curl/inspection).

Tools vs. resources
--------------------
MCP draws a hard line between the two, and this server keeps both sides of
it:

- **Tools** (`tools/list`, `tools/call`) are actions a model *invokes* with
  arguments to produce an effect or a computed result: read this specific
  file, search for a keyword, start watching a directory. The model decides
  when and with what arguments to call them.
- **Resources** (`resources/list`, `resources/read`, `resources/subscribe`)
  are data a *client* enumerates and reads, independent of any model
  decision — here, the resume corpus. A client can list every resume as a
  `file://` URI, read one, or subscribe to be notified when one changes,
  without ever going through a tool call or an LLM turn.

This server backs both with the same directory (see `resume_directory` in
mcp_config.json) but they are handled by entirely separate MCP request
types, registered with separate decorators below.

Error handling: two tiers, by design of the underlying SDK
------------------------------------------------------------
The official Python SDK's low-level `Server.call_tool()` wraps every tool
invocation in a blanket try/except: any exception a tool handler raises
(and any input-schema validation failure) is caught and converted into a
*successful* JSON-RPC response whose `CallToolResult.isError` is `True`,
carrying the exception message as text. This is intentional per the MCP
spec: tool execution failures belong in the result, not the protocol
envelope, so the calling model can see the error and self-correct instead
of the whole request failing. Every tool here follows that path — path
traversal outside `allowed_directories`, a missing file, a file over
`max_file_size_bytes`, and bad tool arguments (caught by the SDK's own
JSON-Schema validation against each tool's `inputSchema`) all surface as
`isError=True` results with a descriptive `"Invalid params: ..."` or
`"Input validation error: ..."` message.

`resources/read`, by contrast, is *not* wrapped that way in the SDK (there
is no result envelope for a resource read to carry an error flag on).
Raising `McpError(ErrorData(code=INVALID_PARAMS, ...))` from the
`read_resource` handler propagates all the way to `_handle_request`, which
converts it into a genuine top-level JSON-RPC error response
(`{"error": {"code": -32602, ...}}`). That path is used here for bad or
out-of-bounds resource URIs — a real protocol-level error, not a
result-level one.

Note on the `resources.subscribe` capability: this SDK version
(mcp==1.29.1) hardcodes `subscribe=False` in the capabilities it reports
during `initialize`, even when a `subscribe_resource` handler is
registered (as this server does). The handler still works if a client
calls `resources/subscribe` directly — this is a capability-advertisement
gap in the SDK, not a functional one. See docs/mcp_architecture.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import aiofiles
from pydantic import AnyUrl
from watchdog.events import FileSystemEventHandler

# The plain (inotify-based) Observer silently never fires on filesystems
# where the kernel doesn't deliver inotify events for external writers —
# notably WSL's DrvFs (/mnt/c/...), and network mounts generally. Polling
# costs a little CPU but works everywhere, which matters more for a
# demoable watch feature than the latency difference.
from watchdog.observers.polling import PollingObserver as Observer

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.shared.exceptions import McpError

import fs_tools

# stdio IS the JSON-RPC transport here: stdout must carry nothing but
# protocol messages, so all logging goes to stderr.
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("filesystem_mcp_server")

SERVER_NAME = "filesystem-mcp"
SERVER_VERSION = "1.0.0"
_DEFAULT_CONFIG_PATH = Path(__file__).parent / "mcp_config.json"
_SUPPORTED_RESUME_EXTENSIONS = {".txt", ".pdf", ".docx"}
_DEFAULT_WATCH_TIMEOUT_SECONDS = 600


# --- Configuration ----------------------------------------------------------


@dataclass
class Config:
    """Validated server configuration, resolved against `base_dir` (the repo root)."""

    allowed_directories: list[str]
    max_file_size_bytes: int
    batch_concurrency: int
    watch_debounce_ms: int
    resume_directory: str
    base_dir: Path

    def resolve(self, path: str) -> str:
        """Resolve `path` (relative to base_dir, or absolute) to a real, symlink-free path."""
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.base_dir / candidate
        return os.path.realpath(candidate)

    def resolved_allowed_dirs(self) -> list[str]:
        return [self.resolve(d) for d in self.allowed_directories]

    def resolved_resume_dir(self) -> str:
        return self.resolve(self.resume_directory)


def _fail(message: str) -> None:
    """Fail loudly on a malformed config: log, print to stderr, and exit non-zero."""
    logger.error("Configuration error: %s", message)
    sys.stderr.write(f"filesystem_mcp_server: configuration error: {message}\n")
    sys.exit(1)


def _env_list(var: str, default: list[str]) -> list[str]:
    raw = os.environ.get(var)
    if raw is None:
        return default
    return [p.strip() for p in raw.split(",") if p.strip()]


def _env_int(var: str, default: int) -> int:
    raw = os.environ.get(var)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        _fail(f"Env var {var}={raw!r} is not a valid integer")
        raise AssertionError("unreachable")  # _fail exits; this satisfies type checkers


def load_config(path: Path) -> Config:
    """Load and validate mcp_config.json, applying MCP_* env var overrides.

    Fails loudly (clear message, non-zero exit) on anything malformed rather
    than starting in a broken state with silently-wrong defaults.
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

    if not isinstance(raw, dict):
        _fail(f"Config file {path} must contain a JSON object at the top level")

    def _require(key: str, expected_type: type) -> Any:
        if key not in raw:
            _fail(f"Config missing required field: {key!r}")
        value = raw[key]
        if not isinstance(value, expected_type):
            _fail(f"Config field {key!r} must be {expected_type.__name__}, got {type(value).__name__}")
        return value

    allowed_directories = _require("allowed_directories", list)
    if not allowed_directories or not all(isinstance(d, str) and d for d in allowed_directories):
        _fail("Config field 'allowed_directories' must be a non-empty list of non-empty strings")
    max_file_size_bytes = _require("max_file_size_bytes", int)
    batch_concurrency = _require("batch_concurrency", int)
    watch_debounce_ms = _require("watch_debounce_ms", int)
    resume_directory = _require("resume_directory", str)

    allowed_directories = _env_list("MCP_ALLOWED_DIRECTORIES", allowed_directories)
    max_file_size_bytes = _env_int("MCP_MAX_FILE_SIZE_BYTES", max_file_size_bytes)
    batch_concurrency = _env_int("MCP_BATCH_CONCURRENCY", batch_concurrency)
    watch_debounce_ms = _env_int("MCP_WATCH_DEBOUNCE_MS", watch_debounce_ms)
    resume_directory = os.environ.get("MCP_RESUME_DIRECTORY", resume_directory)

    if max_file_size_bytes <= 0:
        _fail("Config field 'max_file_size_bytes' must be a positive integer")
    if batch_concurrency <= 0:
        _fail("Config field 'batch_concurrency' must be a positive integer")
    if watch_debounce_ms < 0:
        _fail("Config field 'watch_debounce_ms' must be >= 0")

    return Config(
        allowed_directories=allowed_directories,
        max_file_size_bytes=max_file_size_bytes,
        batch_concurrency=batch_concurrency,
        watch_debounce_ms=watch_debounce_ms,
        resume_directory=resume_directory,
        base_dir=path.parent.resolve(),
    )


CONFIG: Config  # set in main()/init_for_tests() before the server starts handling requests


# --- Security boundary -------------------------------------------------------


class PathSecurityError(ValueError):
    """Raised when a path argument resolves outside every configured allowed_directory."""


def _validate_path(raw_path: str, config: Config) -> str:
    """Resolve `raw_path` and verify it is a descendant of an allowed directory.

    Every filepath/directory argument accepted by any tool passes through
    here before any I/O. This is what closes the path-traversal hole in the
    original fs_tools, where absolute paths and ../ sequences passed
    through to open()/os.listdir() unchecked: os.path.realpath collapses
    both before the containment check, so neither an absolute path outside
    the sandbox nor a relative ../ escape gets through.
    """
    resolved = config.resolve(raw_path)
    for allowed in config.resolved_allowed_dirs():
        if resolved == allowed or resolved.startswith(allowed + os.sep):
            return resolved
    raise PathSecurityError(
        f"Invalid params: path '{raw_path}' resolves to '{resolved}', which is "
        f"outside allowed_directories {config.allowed_directories}"
    )


# --- Watchdog-based filesystem watching --------------------------------------


class _CollectingHandler(FileSystemEventHandler):
    """Debounces watchdog events per (event_type, path) and appends survivors to `queue`.

    Also fires an optional `on_event(event_type, path)` callback — used to
    bridge the always-on resume-directory watcher into MCP notifications
    (watchdog callbacks run on the observer's own thread, not the asyncio
    loop, so that callback is responsible for hopping threads safely).
    """

    def __init__(self, queue: deque, lock: threading.Lock, debounce_ms: int, on_event: Optional[Callable[[str, str], None]] = None) -> None:
        self._queue = queue
        self._lock = lock
        self._debounce_seconds = debounce_ms / 1000.0
        self._last_seen: dict[str, float] = {}
        self._on_event = on_event

    def _handle(self, event_type: str, path: str) -> None:
        if os.path.basename(path).startswith("."):
            return
        now = time.monotonic()
        last = self._last_seen.get(path)
        if last is not None and (now - last) < self._debounce_seconds:
            return
        self._last_seen[path] = now
        with self._lock:
            self._queue.append({"event_type": event_type, "path": path, "timestamp": time.time()})
        if self._on_event is not None:
            try:
                self._on_event(event_type, path)
            except Exception:  # noqa: BLE001
                logger.exception("watch on_event callback failed")

    def on_created(self, event) -> None:  # noqa: ANN001
        if not event.is_directory:
            self._handle("created", event.src_path)

    def on_modified(self, event) -> None:  # noqa: ANN001
        if not event.is_directory:
            self._handle("modified", event.src_path)

    def on_deleted(self, event) -> None:  # noqa: ANN001
        if not event.is_directory:
            self._handle("deleted", event.src_path)


@dataclass
class WatchHandle:
    watch_id: str
    directory: str
    observer: Observer
    events: deque
    lock: threading.Lock
    created_at: float
    timer: Optional[threading.Timer] = None
    stopped: bool = False


_ACTIVE_WATCHES: dict[str, WatchHandle] = {}
_WATCHES_LOCK = threading.Lock()


def _stop_watch(watch_id: str) -> None:
    with _WATCHES_LOCK:
        handle = _ACTIVE_WATCHES.get(watch_id)
    if handle is None or handle.stopped:
        return
    handle.observer.stop()
    handle.observer.join(timeout=2)
    handle.stopped = True
    logger.info("watch %s stopped", watch_id)


def _tool_watch_directory(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    directory = arguments["directory"]
    timeout_seconds = arguments.get("timeout_seconds") or _DEFAULT_WATCH_TIMEOUT_SECONDS
    resolved = _validate_path(directory, config)
    if not os.path.isdir(resolved):
        raise ValueError(f"Invalid params: '{directory}' is not a directory")

    watch_id = uuid.uuid4().hex
    queue: deque = deque()
    lock = threading.Lock()
    handler = _CollectingHandler(queue, lock, config.watch_debounce_ms)
    observer = Observer()
    observer.schedule(handler, resolved, recursive=False)
    observer.start()

    handle = WatchHandle(
        watch_id=watch_id, directory=resolved, observer=observer, events=queue, lock=lock, created_at=time.time()
    )
    timer = threading.Timer(timeout_seconds, _stop_watch, args=(watch_id,))
    timer.daemon = True
    timer.start()
    handle.timer = timer

    with _WATCHES_LOCK:
        _ACTIVE_WATCHES[watch_id] = handle

    logger.info("watch %s started on %s (timeout=%ss)", watch_id, resolved, timeout_seconds)
    return {"watch_id": watch_id, "directory": directory, "status": "watching", "timeout_seconds": timeout_seconds}


def _tool_get_watch_events(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    watch_id = arguments["watch_id"]
    with _WATCHES_LOCK:
        handle = _ACTIVE_WATCHES.get(watch_id)
    if handle is None:
        raise ValueError(f"Invalid params: unknown watch_id '{watch_id}'")
    with handle.lock:
        events = list(handle.events)
        handle.events.clear()
    return {"watch_id": watch_id, "events": events, "active": not handle.stopped}


# --- Filesystem tools (wrapping fs_tools.py) ---------------------------------


def _tool_read_file(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    filepath = arguments["filepath"]
    resolved = _validate_path(filepath, config)
    if os.path.isfile(resolved):
        size = os.path.getsize(resolved)
        if size > config.max_file_size_bytes:
            raise ValueError(
                f"Invalid params: file '{filepath}' is {size} bytes, exceeds "
                f"max_file_size_bytes={config.max_file_size_bytes}"
            )
    result = fs_tools.read_file(resolved)
    if not result.get("success"):
        raise RuntimeError(result.get("error", "read_file failed"))
    return result


def _tool_list_files(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    directory = arguments["directory"]
    extension = arguments.get("extension")
    resolved = _validate_path(directory, config)
    files = fs_tools.list_files(resolved, extension)
    return {"directory": directory, "files": files, "count": len(files)}


async def _tool_write_file(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    """Write text content to a file, via aiofiles rather than fs_tools.write_file.

    Every other tool here delegates to fs_tools and runs it through
    asyncio.to_thread, because pdfplumber/docx parsing has no async
    equivalent to delegate to. A plain UTF-8 text write does, so this is
    the one tool that does real async I/O directly instead of borrowing a
    thread -- replicating fs_tools.write_file's exact contract (mkdir
    parents, same return shape) without going through it.
    """
    filepath = arguments["filepath"]
    content = arguments["content"]
    resolved = _validate_path(filepath, config)
    size = len(content.encode("utf-8"))
    if size > config.max_file_size_bytes:
        raise ValueError(
            f"Invalid params: content is {size} bytes, exceeds max_file_size_bytes={config.max_file_size_bytes}"
        )
    parent = os.path.dirname(resolved)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        async with aiofiles.open(resolved, "w", encoding="utf-8") as fh:
            await fh.write(content)
    except OSError as exc:
        raise RuntimeError(f"Failed to write {filepath}: {exc}") from exc
    return {"success": True, "filepath": resolved, "bytes_written": size}


def _tool_search_in_file(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    filepath = arguments["filepath"]
    keyword = arguments["keyword"]
    resolved = _validate_path(filepath, config)
    result = fs_tools.search_in_file(resolved, keyword)
    if not result.get("success"):
        raise RuntimeError(result.get("error", "search_in_file failed"))
    return result


async def _tool_batch_process(arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    """Process multiple files concurrently, bounded by config.batch_concurrency.

    One bad file never fails the batch: each file's outcome (including
    security-boundary rejections) is captured in its own result entry.
    Reports total_elapsed_seconds alongside a serial-time estimate (the sum
    of each file's own elapsed time) so the concurrency gain is visible.
    """
    filepaths = arguments["filepaths"]
    operation = arguments["operation"]
    if operation not in ("read", "extract_metadata"):
        raise ValueError(f"Invalid params: operation must be 'read' or 'extract_metadata', got {operation!r}")
    if not filepaths:
        raise ValueError("Invalid params: filepaths must be a non-empty list")

    semaphore = asyncio.Semaphore(config.batch_concurrency)

    async def process_one(fp: str) -> dict[str, Any]:
        start = time.monotonic()
        async with semaphore:
            try:
                resolved = _validate_path(fp, config)
            except PathSecurityError as exc:
                return {"filepath": fp, "success": False, "error": str(exc), "elapsed_seconds": round(time.monotonic() - start, 4)}

            if os.path.isfile(resolved) and os.path.getsize(resolved) > config.max_file_size_bytes:
                return {
                    "filepath": fp,
                    "success": False,
                    "error": f"Invalid params: file exceeds max_file_size_bytes={config.max_file_size_bytes}",
                    "elapsed_seconds": round(time.monotonic() - start, 4),
                }

            result = await asyncio.to_thread(fs_tools.read_file, resolved)
            if not result.get("success"):
                return {"filepath": fp, "success": False, "error": result.get("error"), "elapsed_seconds": round(time.monotonic() - start, 4)}

            payload = {"content": result["content"], "metadata": result["metadata"]} if operation == "read" else {"metadata": result["metadata"]}
            return {"filepath": fp, "success": True, **payload, "elapsed_seconds": round(time.monotonic() - start, 4)}

    overall_start = time.monotonic()
    results = await asyncio.gather(*(process_one(fp) for fp in filepaths))
    total_elapsed = time.monotonic() - overall_start
    serial_estimate = sum(r["elapsed_seconds"] for r in results)

    return {
        "operation": operation,
        "results": results,
        "succeeded": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "total_elapsed_seconds": round(total_elapsed, 4),
        "serial_elapsed_estimate_seconds": round(serial_estimate, 4),
        "concurrency": config.batch_concurrency,
    }


# --- Tool registry ------------------------------------------------------------

_TOOLS: list[types.Tool] = [
    types.Tool(
        name="read_file",
        description="Read a .txt, .pdf, or .docx file within an allowed directory; returns text content and metadata.",
        inputSchema={
            "type": "object",
            "properties": {"filepath": {"type": "string", "description": "Path to the file (relative to the repo root, or absolute within an allowed directory)."}},
            "required": ["filepath"],
        },
    ),
    types.Tool(
        name="list_files",
        description="List files in a directory (non-recursive) within an allowed directory, optionally filtered by extension.",
        inputSchema={
            "type": "object",
            "properties": {
                "directory": {"type": "string"},
                "extension": {"type": "string", "description": "Optional filter, e.g. '.pdf' or 'pdf'."},
            },
            "required": ["directory"],
        },
    ),
    types.Tool(
        name="write_file",
        description="Write UTF-8 text content to a file within an allowed directory, creating parent directories as needed.",
        inputSchema={
            "type": "object",
            "properties": {"filepath": {"type": "string"}, "content": {"type": "string"}},
            "required": ["filepath", "content"],
        },
    ),
    types.Tool(
        name="search_in_file",
        description="Case-insensitively search a file's text content for a keyword, returning matches with surrounding context.",
        inputSchema={
            "type": "object",
            "properties": {"filepath": {"type": "string"}, "keyword": {"type": "string"}},
            "required": ["filepath", "keyword"],
        },
    ),
    types.Tool(
        name="watch_directory",
        description=(
            "Start watching a directory for created/modified/deleted files. Returns a watch_id "
            "immediately; poll get_watch_events(watch_id) to drain events detected since the last "
            "poll. The watch stops automatically after timeout_seconds (default 600)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "directory": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1, "description": "Auto-stop after this many seconds (default 600)."},
            },
            "required": ["directory"],
        },
    ),
    types.Tool(
        name="get_watch_events",
        description="Drain filesystem events detected since the last poll for a watch started via watch_directory.",
        inputSchema={
            "type": "object",
            "properties": {"watch_id": {"type": "string"}},
            "required": ["watch_id"],
        },
    ),
    types.Tool(
        name="batch_process",
        description=(
            "Process multiple files concurrently (bounded by the server's batch_concurrency config). "
            "operation is 'read' (full content + metadata) or 'extract_metadata' (metadata only). "
            "Returns per-file results including partial failures, plus timing."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "filepaths": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "operation": {"type": "string", "enum": ["read", "extract_metadata"]},
            },
            "required": ["filepaths", "operation"],
        },
    ),
]

_SYNC_TOOL_DISPATCH: dict[str, Callable[[dict[str, Any], Config], dict[str, Any]]] = {
    "read_file": _tool_read_file,
    "list_files": _tool_list_files,
    "search_in_file": _tool_search_in_file,
    "watch_directory": _tool_watch_directory,
    "get_watch_events": _tool_get_watch_events,
}
# write_file (real async I/O via aiofiles) and batch_process (asyncio.gather)
# are awaited directly rather than run through asyncio.to_thread.
_ASYNC_TOOL_DISPATCH: dict[str, Callable[[dict[str, Any], Config], Any]] = {
    "write_file": _tool_write_file,
    "batch_process": _tool_batch_process,
}


# --- Server-push notification bridge (watchdog thread -> asyncio session) ----

_MAIN_LOOP: Optional[asyncio.AbstractEventLoop] = None
_ACTIVE_SESSION: Optional[Any] = None  # mcp.server.session.ServerSession, captured on first request
_SUBSCRIBED_URIS: set[str] = set()
_SUBSCRIBE_LOCK = threading.Lock()


def _capture_session(server: Server) -> None:
    """Stash the live ServerSession so background threads (watchdog) can push notifications.

    request_context is only valid inside an in-flight request, so this must
    be called from within a request handler; it's a no-op once already
    captured. There's exactly one live session at a time for a stdio
    server (one client per process), so a single module-level slot is fine.
    """
    global _ACTIVE_SESSION
    if _ACTIVE_SESSION is not None:
        return
    try:
        _ACTIVE_SESSION = server.request_context.session
    except LookupError:
        pass


def _notify_resource_list_changed() -> None:
    if _MAIN_LOOP is None or _ACTIVE_SESSION is None:
        return
    asyncio.run_coroutine_threadsafe(_ACTIVE_SESSION.send_resource_list_changed(), _MAIN_LOOP)


def _notify_resource_updated(path: str) -> None:
    if _MAIN_LOOP is None or _ACTIVE_SESSION is None:
        return
    uri = f"file://{path}"
    with _SUBSCRIBE_LOCK:
        subscribed = uri in _SUBSCRIBED_URIS
    if not subscribed:
        return
    asyncio.run_coroutine_threadsafe(_ACTIVE_SESSION.send_resource_updated(AnyUrl(uri)), _MAIN_LOOP)


def _on_resume_dir_event(event_type: str, path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext not in _SUPPORTED_RESUME_EXTENSIONS:
        return
    if event_type in ("created", "deleted"):
        _notify_resource_list_changed()
    elif event_type == "modified":
        _notify_resource_updated(path)


# --- MCP server wiring --------------------------------------------------------

server: Server = Server(SERVER_NAME, version=SERVER_VERSION)


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return _TOOLS


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    _capture_session(server)
    arguments = arguments or {}

    async_func = _ASYNC_TOOL_DISPATCH.get(name)
    if async_func is not None:
        return await async_func(arguments, CONFIG)

    sync_func = _SYNC_TOOL_DISPATCH.get(name)
    if sync_func is None:
        raise ValueError(f"Unknown tool: {name}")
    # Blocking I/O (pdfplumber/docx parsing, os.stat) runs off the event loop.
    return await asyncio.to_thread(sync_func, arguments, CONFIG)


def _mime_type_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".txt": "text/plain",
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }.get(ext, "application/octet-stream")


def _resume_corpus_files(config: Config) -> list[str]:
    resume_dir = config.resolved_resume_dir()
    if not os.path.isdir(resume_dir):
        return []
    return sorted(
        os.path.join(resume_dir, name)
        for name in os.listdir(resume_dir)
        if os.path.isfile(os.path.join(resume_dir, name)) and os.path.splitext(name)[1].lower() in _SUPPORTED_RESUME_EXTENSIONS
    )


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri=AnyUrl(f"file://{path}"),
            name=os.path.basename(path),
            description=f"Resume file: {os.path.basename(path)}",
            mimeType=_mime_type_for(path),
        )
        for path in _resume_corpus_files(CONFIG)
    ]


@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> list[ReadResourceContents]:
    _capture_session(server)
    uri_str = str(uri)
    if not uri_str.startswith("file://"):
        raise McpError(types.ErrorData(code=types.INVALID_PARAMS, message=f"Invalid params: unsupported resource uri scheme: {uri_str}"))

    raw_path = uri_str[len("file://"):]
    try:
        resolved = _validate_path(raw_path, CONFIG)
    except PathSecurityError as exc:
        raise McpError(types.ErrorData(code=types.INVALID_PARAMS, message=str(exc))) from exc

    resume_dir = CONFIG.resolved_resume_dir()
    if resolved != resume_dir and not resolved.startswith(resume_dir + os.sep):
        raise McpError(
            types.ErrorData(code=types.INVALID_PARAMS, message=f"Invalid params: '{raw_path}' is not part of the resume corpus")
        )

    result = await asyncio.to_thread(fs_tools.read_file, resolved)
    if not result.get("success"):
        raise McpError(types.ErrorData(code=types.INVALID_PARAMS, message=f"Invalid params: {result.get('error')}"))

    return [ReadResourceContents(content=result["content"], mime_type=_mime_type_for(resolved))]


@server.subscribe_resource()
async def handle_subscribe_resource(uri: AnyUrl) -> None:
    with _SUBSCRIBE_LOCK:
        _SUBSCRIBED_URIS.add(str(uri))
    logger.info("resource subscribed: %s", uri)


@server.unsubscribe_resource()
async def handle_unsubscribe_resource(uri: AnyUrl) -> None:
    with _SUBSCRIBE_LOCK:
        _SUBSCRIBED_URIS.discard(str(uri))
    logger.info("resource unsubscribed: %s", uri)


# --- Entry point ---------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filesystem MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP transport only")
    parser.add_argument("--port", type=int, default=8765, help="HTTP transport only")
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG_PATH))
    return parser.parse_args(argv)


async def _run_http(host: str, port: int) -> None:
    """Streamable HTTP transport: stateless, JSON responses (not SSE) so curl works directly."""
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Mount

    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    session_manager = StreamableHTTPSessionManager(app=server, json_response=True, stateless=True)

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        async with session_manager.run():
            logger.info("filesystem MCP server (streamable HTTP) listening on http://%s:%s/mcp", host, port)
            yield

    starlette_app = Starlette(routes=[Mount("/mcp", app=session_manager.handle_request)], lifespan=lifespan)
    uvicorn_config = uvicorn.Config(starlette_app, host=host, port=port, log_level="warning")
    await uvicorn.Server(uvicorn_config).serve()


async def main(argv: Optional[list[str]] = None) -> None:
    global CONFIG, _MAIN_LOOP
    args = _parse_args(argv)
    CONFIG = load_config(Path(args.config))
    _MAIN_LOOP = asyncio.get_running_loop()

    resume_dir = CONFIG.resolved_resume_dir()
    resume_observer: Optional[Observer] = None
    if os.path.isdir(resume_dir):
        handler = _CollectingHandler(deque(), threading.Lock(), CONFIG.watch_debounce_ms, on_event=_on_resume_dir_event)
        resume_observer = Observer()
        resume_observer.schedule(handler, resume_dir, recursive=False)
        resume_observer.start()
        logger.info("watching resume_directory %s for resource change notifications", resume_dir)
    else:
        logger.warning("resume_directory %s does not exist; resource change notifications disabled", resume_dir)

    try:
        if args.transport == "stdio":
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options(
                        notification_options=NotificationOptions(resources_changed=True, tools_changed=False, prompts_changed=False)
                    ),
                )
        else:
            await _run_http(args.host, args.port)
    finally:
        if resume_observer is not None:
            resume_observer.stop()
            resume_observer.join(timeout=2)
        for watch_id in list(_ACTIVE_WATCHES):
            _stop_watch(watch_id)


if __name__ == "__main__":
    asyncio.run(main())
