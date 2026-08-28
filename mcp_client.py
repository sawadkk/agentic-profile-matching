"""Connection manager for MCP servers -- the agent's only path to the filesystem.

matching_agent.py no longer imports fs_tools directly. Every filesystem
operation (and every notes-server operation) it needs goes through
`MCPClientManager`, which launches the configured MCP servers as
subprocesses, discovers their tools and resources at startup (never
hardcoded), and hands LangGraph a unified list of LangChain tools drawn
from every server.

Subprocess lifecycle, honestly stated
--------------------------------------
`langchain-mcp-adapters` 0.3.x's `MultiServerMCPClient` opens a fresh stdio
subprocess for each `tools/list` / `resources/list` call and for each
individual tool invocation, tearing it down when that call returns (this is
documented behavior of `MultiServerMCPClient.get_tools()`, verified
directly against the installed package here). That means there is no
persistent subprocess for this module to leak or orphan -- each call is
already scoped. What `MCPClientManager` actually owns is the one-time
startup discovery handshake (reading mcp_config.json, launching each server
briefly to enumerate its tools/resources, and failing fast with a clear
error if a server can't be reached) and the resulting tool cache. Its
`async with` / `initialize()` contract exists for that discovery step and
for a clean place to fail loudly on a misconfigured or missing server, not
because it is holding an open connection between calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.resolve()
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "mcp_config.json"


class MCPClientError(Exception):
    """Raised when an MCP server is unavailable or misconfigured, with a clear message."""


def _load_server_specs(config_path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MCPClientError(f"MCP config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise MCPClientError(f"MCP config file {config_path} is not valid JSON: {exc}") from exc

    servers = raw.get("servers") if isinstance(raw, dict) else None
    if not isinstance(servers, dict) or not servers:
        raise MCPClientError(f"MCP config file {config_path} has no non-empty 'servers' section")
    return servers


def _build_connections(servers: dict[str, Any], config_path: Path) -> dict[str, StdioConnection]:
    connections: dict[str, StdioConnection] = {}
    for name, spec in servers.items():
        command = spec.get("command") if isinstance(spec, dict) else None
        args = spec.get("args") if isinstance(spec, dict) else None
        if not command or not isinstance(args, list):
            raise MCPClientError(f"MCP server '{name}' config must have string 'command' and list 'args'")
        connections[name] = StdioConnection(
            transport="stdio",
            command=command,
            args=[str(a) for a in args],
            cwd=str(config_path.parent),
            env={**os.environ, "MCP_CONFIG_PATH": str(config_path)},
        )
    return connections


def _parse_tool_result(raw: Any) -> dict[str, Any]:
    """Unwrap a LangChain MCP tool's return value into a plain status dict.

    `.ainvoke()` on an MCP-backed tool returns a list of LangChain content
    blocks, e.g. [{"type": "text", "text": "..."}]. On success the text is
    the tool's JSON payload (our servers' tools all return dicts, which the
    SDK JSON-serializes); on an MCP-level error (isError=True, or an
    input-schema validation failure) it's a plain message, not JSON. Either
    way this returns a dict so callers can keep treating MCP tools exactly
    like fs_tools.py's own never-raises, always-a-status-dict functions.
    """
    if isinstance(raw, dict):
        return raw

    text_parts: list[str] = []
    if isinstance(raw, list):
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
    elif isinstance(raw, str):
        text_parts.append(raw)
    text = "\n".join(text_parts) if text_parts else str(raw)

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"success": False, "error": text}

    return parsed if isinstance(parsed, dict) else {"success": True, "result": parsed}


class MCPClientManager:
    """Discovers tools/resources from every configured MCP server and calls them.

    Usage as a scoped context manager (tests, standalone scripts):
        async with MCPClientManager() as mcp:
            result = await mcp.call_tool("read_file", filepath="...")

    Usage as a long-lived singleton (what matching_agent.py does, via
    `get_manager()`): call `await manager.initialize()` once per process
    and keep using it across many agent turns -- see the module docstring
    for why there's no persistent resource that needs closing between
    calls.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        self._client: Optional[MultiServerMCPClient] = None
        self._tools_by_name: dict[str, BaseTool] = {}
        self._tool_server: dict[str, str] = {}
        self._resources_by_server: dict[str, list[Any]] = {}
        self._server_names: list[str] = []
        self._unavailable_servers: dict[str, str] = {}
        self._initialized = False

    @property
    def unavailable_servers(self) -> dict[str, str]:
        """server_name -> error message, for servers that failed to launch/respond."""
        return dict(self._unavailable_servers)

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools_by_name.values())

    @property
    def server_names(self) -> list[str]:
        return list(self._server_names)

    @property
    def resources_by_server(self) -> dict[str, list[Any]]:
        return dict(self._resources_by_server)

    def server_for_tool(self, tool_name: str) -> Optional[str]:
        """The server a discovered tool came from, e.g. 'filesystem' for 'read_file'."""
        return self._tool_server.get(tool_name)

    def trace_label(self, tool_name: str) -> str:
        """Reasoning-trace label for a tool call, e.g. 'mcp:filesystem/read_file'."""
        server_name = self._tool_server.get(tool_name, "unknown")
        return f"mcp:{server_name}/{tool_name}"

    async def initialize(self) -> None:
        """Discover tools and resources from every configured server. Idempotent.

        Raises MCPClientError only for a broken *configuration* -- a
        missing/malformed mcp_config.json, or a server entry missing
        'command'/'args'; that's a config-authoring bug worth failing
        loudly on. A server that's configured correctly but fails to
        actually launch or respond (wrong script path, non-zero exit,
        interpreter missing) is handled per-server instead: it's recorded
        in `unavailable_servers` and skipped, so one broken server doesn't
        take down tools from servers that DO work. `call_tool` surfaces the
        specific reason for an unavailable tool rather than a bare
        "unknown tool".
        """
        if self._initialized:
            return

        servers = _load_server_specs(self._config_path)
        connections = _build_connections(servers, self._config_path)
        self._client = MultiServerMCPClient(connections)
        self._server_names = list(connections)

        for server_name in self._server_names:
            spec = connections[server_name]
            try:
                tools = await self._client.get_tools(server_name=server_name)
            except Exception as exc:  # noqa: BLE001
                message = (
                    f"MCP server '{server_name}' is unavailable (launch command: "
                    f"{spec['command']} {' '.join(spec['args'])}, cwd={spec['cwd']}). "
                    f"Check that mcp_config.json's 'servers' entry is correct and the "
                    f"script is runnable. Underlying error: {type(exc).__name__}: {exc}"
                )
                logger.warning(message)
                self._unavailable_servers[server_name] = message
                continue

            for tool in tools:
                if tool.name in self._tools_by_name and self._tool_server[tool.name] != server_name:
                    logger.warning(
                        "Tool name collision: '%s' offered by both '%s' and '%s'; keeping '%s'",
                        tool.name, self._tool_server[tool.name], server_name, server_name,
                    )
                self._tools_by_name[tool.name] = tool
                self._tool_server[tool.name] = server_name

            try:
                # Deliberately session.list_resources() (metadata only), not
                # MultiServerMCPClient.get_resources() -- that helper reads
                # the full content of every resource via resources/read to
                # build LangChain Blobs, which for a 100-file resume corpus
                # means 100 file reads (PDF/DOCX parsing included) on every
                # startup. Discovery only needs the listing.
                async with self._client.session(server_name) as session:
                    listed = await session.list_resources()
                self._resources_by_server[server_name] = [
                    {"uri": str(r.uri), "name": r.name, "mimeType": r.mimeType} for r in listed.resources
                ]
            except Exception as exc:  # noqa: BLE001
                # Resources are optional per-server (notes has none); don't fail startup over it.
                logger.debug("Server '%s' has no resources or resources/list failed: %s", server_name, exc)
                self._resources_by_server[server_name] = []

            logger.info(
                "MCP server '%s': discovered %d tools (%s), %d resources",
                server_name, len(tools), [t.name for t in tools], len(self._resources_by_server[server_name]),
            )

        self._initialized = True

    async def call_tool(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        """Call a discovered MCP tool by name and return a uniform status dict.

        Never raises for an MCP-level failure (unknown tool, tool execution
        error, malformed response, or a transport/subprocess failure) --
        those all come back as {"success": False, "error": str}, matching
        fs_tools.py's own never-raises convention, so graph nodes handle an
        MCP failure exactly like they always handled an fs_tools failure.
        """
        await self.initialize()
        tool = self._tools_by_name.get(tool_name)
        if tool is None:
            if self._unavailable_servers:
                return {
                    "success": False,
                    "error": (
                        f"Unknown MCP tool: '{tool_name}'. Unavailable server(s): "
                        f"{'; '.join(self._unavailable_servers.values())}"
                    ),
                }
            return {
                "success": False,
                "error": f"Unknown MCP tool: '{tool_name}' (known: {sorted(self._tools_by_name)})",
            }

        try:
            raw = await tool.ainvoke(kwargs)
        except Exception as exc:  # noqa: BLE001
            server_name = self._tool_server.get(tool_name, "?")
            logger.warning("MCP call '%s' on server '%s' failed: %s", tool_name, server_name, exc)
            return {"success": False, "error": f"MCP call to '{tool_name}' failed: {exc}"}

        return _parse_tool_result(raw)

    async def __aenter__(self) -> "MCPClientManager":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # See the class docstring: there is no persistent subprocess held
        # between calls, so there's nothing to tear down beyond dropping
        # our own cache. This exists so callers get a normal context-manager
        # shape (and so re-entering `async with` after this starts clean).
        self._client = None
        self._tools_by_name.clear()
        self._tool_server.clear()
        self._resources_by_server.clear()
        self._unavailable_servers.clear()
        self._initialized = False


# --- Process-wide singleton, used by matching_agent.py -----------------------

_singleton: Optional[MCPClientManager] = None
_singleton_lock = asyncio.Lock()


async def get_manager() -> MCPClientManager:
    """Return the process-wide MCPClientManager, initializing it on first use."""
    global _singleton
    if _singleton is None:
        async with _singleton_lock:
            if _singleton is None:
                _singleton = MCPClientManager()
    await _singleton.initialize()
    return _singleton


async def _describe() -> None:
    """Standalone inspection: `python mcp_client.py` prints what's discovered."""
    logging.basicConfig(level=logging.INFO)
    async with MCPClientManager() as manager:
        for server_name in manager.server_names:
            print(f"\n=== {server_name} ===")
            tool_names = [t.name for t in manager.tools if manager.server_for_tool(t.name) == server_name]
            print(f"tools: {tool_names}")
            resources = manager.resources_by_server.get(server_name, [])
            print(f"resources: {len(resources)}" + (f" (first: {resources[0]})" if resources else ""))


if __name__ == "__main__":
    asyncio.run(_describe())
