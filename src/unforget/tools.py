"""Memory tools for LLM agents — provider-agnostic schemas + executor.

Gives LLMs 5 tools to manage their own memory:
  memory_store  — remember a fact
  memory_search — recall relevant memories
  memory_list   — browse memories by type/tags
  memory_forget — remove a memory
  memory_update — supersede a memory with updated content

Works with any LLM provider. Formatters convert to OpenAI/Anthropic schemas.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from unforget.store import MemoryStore

logger = logging.getLogger("unforget.tools")

# ---------------------------------------------------------------------------
# Tool definitions (provider-agnostic)
# ---------------------------------------------------------------------------

MEMORY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "memory_store",
        "description": (
            "Store a fact, preference, or important information for future conversations. "
            "Use this when you learn something worth remembering about the user, their project, "
            "or their preferences. Be concise and specific."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The fact to remember. Be concise and specific.",
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["insight", "event"],
                    "default": "insight",
                    "description": "insight = distilled fact/preference. event = specific interaction.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Categorization tags. Always include at least one from: "
                        "persona (name, role, identity), preference (likes, dislikes, style), "
                        "skill (languages, tools, expertise), project (current work, goals), "
                        "relationship (people, teams, orgs), decision (choices made and why), "
                        "procedural (rules, workflows, habits), emotional (mood, feelings), "
                        "context (environment, setup, config). Add specific sub-tags too."
                    ),
                },
                "importance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                    "description": "How important is this memory? 0.0 = trivial, 0.5 = normal, 1.0 = critical. Higher importance memories surface first in recall.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "memory_search",
        "description": (
            "Search your memory for relevant information. Use this when you need context "
            "about the user, their project, or past interactions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for.",
                },
                "limit": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum number of results.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_list",
        "description": "List stored memories, optionally filtered by type or tags.",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_type": {
                    "type": "string",
                    "enum": ["insight", "event", "raw"],
                    "description": "Filter by memory type.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by tags (any match).",
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Maximum number of results.",
                },
            },
        },
    },
    {
        "name": "memory_forget",
        "description": "Remove a memory that is no longer true or relevant.",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "ID of the memory to forget (from memory_search or memory_list).",
                },
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "memory_update",
        "description": (
            "Update a memory with new information. The old version is preserved in history. "
            "Use this when a previously stored fact has changed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "ID of the memory to update.",
                },
                "new_content": {
                    "type": "string",
                    "description": "The updated fact.",
                },
            },
            "required": ["memory_id", "new_content"],
        },
    },
]

MEMORY_SYSTEM_INSTRUCTION = (
    "\n\n[Memory Instructions]\n"
    "You have access to persistent memory that survives across conversations. "
    "Use memory_store to save important facts about the user, their preferences, "
    "and decisions. Use memory_search when you need context from past interactions. "
    "Use memory_update when a fact changes. Use memory_forget to remove outdated info. "
    "Memory is your long-term knowledge — store insights, not transcripts."
)


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

class MemoryToolExecutor:
    """Executes memory tool calls from any LLM provider.

    Usage::

        executor = MemoryToolExecutor(store, org_id="acme", agent_id="bot")

        # Get tool schemas
        openai_tools = executor.to_openai()
        anthropic_tools = executor.to_anthropic()

        # Execute a tool call
        result = await executor.execute("memory_store", {"content": "User prefers Fly.io"})

        # Handle full LLM response
        tool_messages = await executor.handle_openai_response(response)
        tool_messages = await executor.handle_anthropic_response(response)
    """

    def __init__(
        self,
        store: MemoryStore,
        org_id: str,
        agent_id: str,
        *,
        tools: list[str] | None = None,
    ):
        self.store = store
        self.org_id = org_id
        self.agent_id = agent_id
        # Allow selecting a subset of tools
        self._enabled = set(tools) if tools else {t["name"] for t in MEMORY_TOOLS}

    @property
    def tool_names(self) -> set[str]:
        return self._enabled

    def _get_tools(self) -> list[dict]:
        return [t for t in MEMORY_TOOLS if t["name"] in self._enabled]

    # -------------------------------------------------------------------
    # Schema formatters
    # -------------------------------------------------------------------

    def to_openai(self) -> list[dict]:
        """Convert to OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in self._get_tools()
        ]

    def to_anthropic(self) -> list[dict]:
        """Convert to Anthropic tool_use format."""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in self._get_tools()
        ]

    def to_generic(self) -> list[dict]:
        """Return the raw provider-agnostic tool definitions."""
        return self._get_tools()

    # -------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a memory tool call. Returns result as string for the LLM."""
        if tool_name not in self._enabled:
            return f"Error: unknown tool '{tool_name}'"

        try:
            if tool_name == "memory_store":
                return await self._exec_store(arguments)
            elif tool_name == "memory_search":
                return await self._exec_search(arguments)
            elif tool_name == "memory_list":
                return await self._exec_list(arguments)
            elif tool_name == "memory_forget":
                return await self._exec_forget(arguments)
            elif tool_name == "memory_update":
                return await self._exec_update(arguments)
            else:
                return f"Error: unhandled tool '{tool_name}'"
        except Exception as e:
            logger.warning("Tool %s failed: %s", tool_name, e)
            return f"Error: {e}"

    async def _exec_store(self, args: dict) -> str:
        item = await self.store.write(
            args["content"],
            org_id=self.org_id,
            agent_id=self.agent_id,
            memory_type=args.get("memory_type", "insight"),
            tags=args.get("tags", []),
            importance=args.get("importance", 0.5),
        )
        return f"Stored: \"{item.content}\" (id: {item.id})"

    async def _exec_search(self, args: dict) -> str:
        results = await self.store.recall(
            args["query"],
            org_id=self.org_id,
            agent_id=self.agent_id,
            limit=args.get("limit", 5),
        )
        if not results:
            return "No memories found."
        lines = []
        for r in results:
            lines.append(f"- [{r.id}] {r.content} (type: {r.memory_type.value}, score: {r.score:.2f})")
        return "Found memories:\n" + "\n".join(lines)

    async def _exec_list(self, args: dict) -> str:
        mt = args.get("memory_type")
        items = await self.store.list(
            org_id=self.org_id,
            agent_id=self.agent_id,
            memory_type=mt,
            tags=args.get("tags"),
            page_size=args.get("limit", 20),
        )
        if not items:
            return "No memories stored."
        lines = []
        for m in items:
            tags_str = f" tags={m.tags}" if m.tags else ""
            lines.append(f"- [{m.id}] {m.content} (type: {m.memory_type.value}{tags_str})")
        return f"{len(items)} memories:\n" + "\n".join(lines)

    async def _exec_forget(self, args: dict) -> str:
        try:
            mid = uuid.UUID(args["memory_id"])
        except ValueError:
            return f"Error: invalid memory_id '{args['memory_id']}'"
        success = await self.store.forget(mid)
        return "Memory deleted." if success else "Memory not found."

    async def _exec_update(self, args: dict) -> str:
        try:
            mid = uuid.UUID(args["memory_id"])
        except ValueError:
            return f"Error: invalid memory_id '{args['memory_id']}'"
        try:
            old, new = await self.store.supersede(
                mid,
                args["new_content"],
                org_id=self.org_id,
                agent_id=self.agent_id,
            )
            return f"Updated: \"{old.content}\" → \"{new.content}\" (new id: {new.id})"
        except ValueError as e:
            return f"Error: {e}"

    # -------------------------------------------------------------------
    # Response handlers
    # -------------------------------------------------------------------

    async def handle_openai_response(self, response: Any) -> list[dict]:
        """Process tool_calls from an OpenAI ChatCompletion response.

        Returns a list of tool result messages to append to the conversation.
        Only handles memory_* tool calls — passes through others.
        """
        messages = []
        choice = response.choices[0] if response.choices else None
        if not choice or not choice.message.tool_calls:
            return messages

        for tc in choice.message.tool_calls:
            if tc.function.name not in self._enabled:
                continue
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result = await self.execute(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        return messages

    async def handle_anthropic_response(self, response: Any) -> list[dict]:
        """Process tool_use blocks from an Anthropic Messages response.

        Returns a list of tool_result content blocks for the next user message.
        Only handles memory_* tool calls.
        """
        results = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            if block.name not in self._enabled:
                continue
            result = await self.execute(block.name, block.input)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        return results

    async def handle_openai_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        """Execute tools from accumulated streaming deltas (parsed dicts).

        Args:
            tool_calls: List of dicts with 'id', 'function.name', 'function.arguments'.

        Returns:
            List of tool result messages for the conversation.
        """
        messages = []
        for tc in tool_calls:
            name = tc["function"]["name"]
            if name not in self._enabled:
                continue
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}
            result = await self.execute(name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })
        return messages

    async def handle_anthropic_tool_blocks(self, tool_blocks: list[dict]) -> list[dict]:
        """Execute tools from accumulated streaming blocks (parsed dicts).

        Args:
            tool_blocks: List of dicts with 'id', 'name', 'input'.

        Returns:
            List of tool_result content blocks for the next user message.
        """
        results = []
        for block in tool_blocks:
            if block.get("type") != "tool_use":
                continue
            name = block["name"]
            if name not in self._enabled:
                continue
            result = await self.execute(name, block["input"])
            results.append({
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": result,
            })
        return results

    def is_memory_tool_call(self, tool_name: str) -> bool:
        """Check if a tool name is a memory tool."""
        return tool_name in self._enabled
