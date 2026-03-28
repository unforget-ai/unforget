"""Anthropic SDK integration for unforget.

Wraps AsyncAnthropic to transparently inject memory into every message call.

Usage::

    from anthropic import AsyncAnthropic
    from unforget import MemoryStore
    from unforget.integrations.anthropic import wrap_anthropic

    store = MemoryStore("postgresql://...")
    await store.initialize()

    client = wrap_anthropic(AsyncAnthropic(), store, org_id="acme", agent_id="bot")
    response = await client.messages.create(
        messages=[{"role": "user", "content": "deploy the API"}],
        model="claude-sonnet-4-6",
        max_tokens=1024,
    )

Install: pip install unforget[anthropic]
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from unforget.integrations.streaming import (
    AnthropicToolAccumulator,
    background_ingest,
)
from unforget.store import MemoryStore
from unforget.tools import MEMORY_SYSTEM_INSTRUCTION, MemoryToolExecutor

logger = logging.getLogger("unforget.integrations.anthropic")


class MemoryMessages:
    """Wraps an Anthropic client's messages interface with memory."""

    def __init__(
        self,
        client: Any,
        executor: MemoryToolExecutor,
        store: MemoryStore,
        org_id: str,
        agent_id: str,
        *,
        auto_recall: bool = True,
        auto_ingest: bool = True,
        inject_tools: bool = True,
        inject_instructions: bool = True,
    ):
        self._client = client
        self._executor = executor
        self._store = store
        self._org_id = org_id
        self._agent_id = agent_id
        self._auto_recall = auto_recall
        self._auto_ingest = auto_ingest
        self._inject_tools = inject_tools
        self._inject_instructions = inject_instructions

    async def create(self, *, messages: list[dict], **kwargs) -> Any:
        """Create a message with memory integration.

        When stream=True, returns an AnthropicMemoryStream (async iterator).
        Otherwise returns a Message response (same as before).
        """
        if kwargs.get("stream"):
            return await self._create_stream(messages=messages, **kwargs)
        return await self._create_sync(messages=messages, **kwargs)

    async def _create_sync(self, *, messages: list[dict], **kwargs) -> Any:
        """Non-streaming message with memory (original path)."""
        messages = list(messages)

        # Step 1: Build system prompt additions
        system_additions = ""

        if self._auto_recall:
            user_msg = _find_last_user_message(messages)
            if user_msg:
                context = await self._store.auto_recall(
                    user_msg, org_id=self._org_id, agent_id=self._agent_id,
                )
                if context:
                    system_additions += "\n\n" + context

        if self._inject_instructions:
            system_additions += MEMORY_SYSTEM_INSTRUCTION

        # Anthropic uses `system` as a top-level kwarg, not in messages
        if system_additions:
            existing_system = kwargs.get("system", "")
            kwargs["system"] = existing_system + system_additions

        # Step 2: Add memory tools
        if self._inject_tools:
            existing_tools = kwargs.get("tools", []) or []
            memory_tools = self._executor.to_anthropic()
            kwargs["tools"] = existing_tools + memory_tools

        # Step 3: Call Anthropic
        response = await self._client.messages.create(messages=messages, **kwargs)

        # Step 4: Handle memory tool calls
        if self._inject_tools:
            response, messages = await self._handle_tool_loop(
                response, messages, **kwargs
            )

        # Step 5: Auto-ingest
        if self._auto_ingest and messages:
            try:
                await self._store.ingest(
                    messages,
                    org_id=self._org_id,
                    agent_id=self._agent_id,
                    mode="background",
                )
            except Exception as e:
                logger.debug("Auto-ingest failed (non-fatal): %s", e)

        return response

    async def _handle_tool_loop(
        self, response: Any, messages: list[dict], **kwargs
    ) -> tuple[Any, list[dict]]:
        """Process memory tool_use blocks, looping until LLM stops."""
        max_rounds = 5

        for _ in range(max_rounds):
            # Check for tool_use blocks
            tool_use_blocks = [
                b for b in response.content
                if getattr(b, "type", None) == "tool_use"
                and self._executor.is_memory_tool_call(b.name)
            ]
            if not tool_use_blocks:
                break

            # Add assistant response to messages
            messages.append({
                "role": "assistant",
                "content": [_block_to_dict(b) for b in response.content],
            })

            # Execute memory tool calls and build tool_result message
            tool_results = await self._executor.handle_anthropic_response(response)
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Call Anthropic again
            response = await self._client.messages.create(
                messages=messages, **kwargs
            )

        return response, messages


    async def _create_stream(self, *, messages: list[dict], **kwargs) -> AnthropicMemoryStream:
        """Streaming message with memory support."""
        messages = list(messages)

        # Build system prompt additions
        system_additions = ""
        if self._auto_recall:
            user_msg = _find_last_user_message(messages)
            if user_msg:
                context = await self._store.auto_recall(
                    user_msg, org_id=self._org_id, agent_id=self._agent_id,
                )
                if context:
                    system_additions += "\n\n" + context

        if self._inject_instructions:
            system_additions += MEMORY_SYSTEM_INSTRUCTION

        if system_additions:
            existing_system = kwargs.get("system", "")
            kwargs["system"] = existing_system + system_additions

        if self._inject_tools:
            existing_tools = kwargs.get("tools", []) or []
            memory_tools = self._executor.to_anthropic()
            kwargs["tools"] = existing_tools + memory_tools

        stream = await self._client.messages.create(messages=messages, **kwargs)

        return AnthropicMemoryStream(
            stream=stream,
            client=self._client,
            executor=self._executor,
            store=self._store,
            org_id=self._org_id,
            agent_id=self._agent_id,
            messages=messages,
            inject_tools=self._inject_tools,
            auto_ingest=self._auto_ingest,
            kwargs=kwargs,
        )


class AnthropicMemoryStream:
    """Async iterator that wraps an Anthropic stream with memory tool handling.

    Yields Anthropic stream events (message_start, content_block_delta, etc.).
    If the message ends with tool_use blocks, executes memory tools and starts
    a new stream transparently.
    """

    def __init__(
        self,
        stream: Any,
        client: Any,
        executor: MemoryToolExecutor,
        store: MemoryStore,
        org_id: str,
        agent_id: str,
        messages: list[dict],
        inject_tools: bool,
        auto_ingest: bool,
        kwargs: dict,
    ):
        self._stream = stream
        self._client = client
        self._executor = executor
        self._store = store
        self._org_id = org_id
        self._agent_id = agent_id
        self._messages = messages
        self._inject_tools = inject_tools
        self._auto_ingest = auto_ingest
        self._kwargs = kwargs
        self._accumulator = AnthropicToolAccumulator()
        self._rounds = 0
        self._max_rounds = 5
        self._finished = False
        self._stop_reason: str | None = None

    def __aiter__(self):
        return self

    async def __anext__(self) -> Any:
        while True:
            try:
                event = await self._stream.__anext__()
                self._accumulator.accumulate(event)

                # Track stop reason from message_delta events
                if getattr(event, "type", None) == "message_delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        sr = getattr(delta, "stop_reason", None)
                        if sr:
                            self._stop_reason = sr

                return event
            except StopAsyncIteration:
                if await self._handle_stream_end():
                    continue
                raise

    async def _handle_stream_end(self) -> bool:
        """Handle end of a stream. Returns True if a new stream was started."""
        tool_blocks = self._accumulator.get_tool_blocks()

        if (
            not tool_blocks
            or not self._inject_tools
            or self._rounds >= self._max_rounds
            or self._stop_reason != "tool_use"
        ):
            if self._auto_ingest and self._messages:
                asyncio.create_task(
                    background_ingest(
                        self._store, self._messages,
                        self._org_id, self._agent_id,
                    )
                )
            self._finished = True
            return False

        memory_blocks = [
            b for b in tool_blocks
            if self._executor.is_memory_tool_call(b["name"])
        ]
        if not memory_blocks:
            if self._auto_ingest and self._messages:
                asyncio.create_task(
                    background_ingest(
                        self._store, self._messages,
                        self._org_id, self._agent_id,
                    )
                )
            self._finished = True
            return False

        # Add assistant message with all content blocks
        self._messages.append({
            "role": "assistant",
            "content": tool_blocks,
        })

        # Execute memory tool calls
        tool_results = await self._executor.handle_anthropic_tool_blocks(tool_blocks)
        if tool_results:
            self._messages.append({"role": "user", "content": tool_results})

        # Start new stream
        self._accumulator.reset()
        self._stop_reason = None
        self._rounds += 1
        self._stream = await self._client.messages.create(
            messages=self._messages, **self._kwargs
        )
        return True


class WrappedAnthropic:
    """Proxy for AsyncAnthropic that adds memory to messages."""

    def __init__(self, client: Any, memory_messages: MemoryMessages):
        self._client = client
        self._memory_messages = memory_messages

    @property
    def messages(self):
        return self._memory_messages

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def wrap_anthropic(
    client: Any,
    store: MemoryStore,
    org_id: str,
    agent_id: str,
    *,
    auto_recall: bool = True,
    auto_ingest: bool = True,
    inject_tools: bool = True,
    inject_instructions: bool = True,
    tools: list[str] | None = None,
) -> WrappedAnthropic:
    """Wrap an AsyncAnthropic client with transparent memory.

    Args:
        client: An AsyncAnthropic() instance.
        store: Initialized MemoryStore.
        org_id: Organization scope for memories.
        agent_id: Agent scope for memories.
        auto_recall: Inject relevant memories into system prompt.
        auto_ingest: Store conversation after response.
        inject_tools: Add memory tools to every call.
        inject_instructions: Add memory usage instructions to system prompt.
        tools: Subset of memory tools to enable (default: all 5).
    """
    executor = MemoryToolExecutor(store, org_id, agent_id, tools=tools)
    memory_messages = MemoryMessages(
        client,
        executor,
        store,
        org_id,
        agent_id,
        auto_recall=auto_recall,
        auto_ingest=auto_ingest,
        inject_tools=inject_tools,
        inject_instructions=inject_instructions,
    )
    return WrappedAnthropic(client, memory_messages)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_last_user_message(messages: list[dict]) -> str | None:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
    return None


def _block_to_dict(block: Any) -> dict:
    """Convert an Anthropic content block to a dict."""
    if hasattr(block, "model_dump"):
        return block.model_dump()
    if hasattr(block, "type"):
        if block.type == "text":
            return {"type": "text", "text": block.text}
        elif block.type == "tool_use":
            return {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            }
    return {"type": "text", "text": str(block)}
