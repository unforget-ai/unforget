"""OpenAI SDK integration for unforget.

Wraps AsyncOpenAI to transparently inject memory into every chat completion.

Usage::

    from openai import AsyncOpenAI
    from unforget import MemoryStore
    from unforget.integrations.openai import wrap_openai

    store = MemoryStore("postgresql://...")
    await store.initialize()

    client = wrap_openai(AsyncOpenAI(), store, org_id="acme", agent_id="bot")
    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": "deploy the API"}],
        model="gpt-4o",
    )
    # Memory handled transparently.

Install: pip install unforget[openai]
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from unforget.integrations.streaming import (
    OpenAIToolAccumulator,
    background_ingest,
)
from unforget.store import MemoryStore
from unforget.tools import MEMORY_SYSTEM_INSTRUCTION, MemoryToolExecutor

logger = logging.getLogger("unforget.integrations.openai")


class MemoryChat:
    """Wraps an OpenAI client's chat.completions interface with memory."""

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

    @property
    def completions(self):
        return self

    async def create(self, *, messages: list[dict], **kwargs) -> Any:
        """Create a chat completion with memory integration.

        When stream=True, returns an OpenAIMemoryStream (async iterator).
        Otherwise returns a ChatCompletion response (same as before).
        """
        if kwargs.get("stream"):
            return await self._create_stream(messages=messages, **kwargs)
        return await self._create_sync(messages=messages, **kwargs)

    async def _create_sync(self, *, messages: list[dict], **kwargs) -> Any:
        """Non-streaming chat completion with memory (original path)."""
        messages = list(messages)

        # Step 1: Auto-recall — inject memory context into system prompt
        if self._auto_recall:
            user_msg = _find_last_user_message(messages)
            if user_msg:
                context = await self._store.auto_recall(
                    user_msg, org_id=self._org_id, agent_id=self._agent_id,
                )
                if context:
                    messages = _inject_system_context(messages, context)

        # Step 2: Inject memory instructions into system prompt
        if self._inject_instructions:
            messages = _inject_system_context(messages, MEMORY_SYSTEM_INSTRUCTION)

        # Step 3: Add memory tools
        if self._inject_tools:
            existing_tools = kwargs.get("tools", []) or []
            memory_tools = self._executor.to_openai()
            kwargs["tools"] = existing_tools + memory_tools

        # Step 4: Call OpenAI
        response = await self._client.chat.completions.create(
            messages=messages, **kwargs
        )

        # Step 5: Handle memory tool calls (loop until no more tool calls)
        if self._inject_tools:
            response, messages = await self._handle_tool_loop(
                response, messages, **kwargs
            )

        # Step 6: Auto-ingest conversation
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
        """Process memory tool calls, looping until the LLM stops calling tools."""
        max_rounds = 5

        for _ in range(max_rounds):
            choice = response.choices[0] if response.choices else None
            if not choice or not choice.message.tool_calls:
                break

            # Check if any tool calls are memory tools
            memory_calls = [
                tc for tc in choice.message.tool_calls
                if self._executor.is_memory_tool_call(tc.function.name)
            ]
            if not memory_calls:
                break

            # Add assistant message with tool calls
            messages.append(choice.message.model_dump())

            # Execute memory tool calls
            tool_results = await self._executor.handle_openai_response(response)
            messages.extend(tool_results)

            # Call LLM again with tool results
            response = await self._client.chat.completions.create(
                messages=messages, **kwargs
            )

        return response, messages


    async def _create_stream(self, *, messages: list[dict], **kwargs) -> OpenAIMemoryStream:
        """Streaming chat completion with memory support."""
        messages = list(messages)

        # Auto-recall (runs upfront, ~80ms)
        if self._auto_recall:
            user_msg = _find_last_user_message(messages)
            if user_msg:
                context = await self._store.auto_recall(
                    user_msg, org_id=self._org_id, agent_id=self._agent_id,
                )
                if context:
                    messages = _inject_system_context(messages, context)

        if self._inject_instructions:
            messages = _inject_system_context(messages, MEMORY_SYSTEM_INSTRUCTION)

        if self._inject_tools:
            existing_tools = kwargs.get("tools", []) or []
            memory_tools = self._executor.to_openai()
            kwargs["tools"] = existing_tools + memory_tools

        stream = await self._client.chat.completions.create(
            messages=messages, **kwargs
        )

        return OpenAIMemoryStream(
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


class OpenAIMemoryStream:
    """Async iterator that wraps an OpenAI stream with memory tool handling.

    Yields ChatCompletionChunk objects. If the stream ends with tool calls,
    executes memory tools and starts a new stream transparently.
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
        self._accumulator = OpenAIToolAccumulator()
        self._rounds = 0
        self._max_rounds = 5
        self._finished = False
        self._finish_reason: str | None = None

    def __aiter__(self):
        return self

    async def __anext__(self) -> Any:
        while True:
            try:
                chunk = await self._stream.__anext__()
                self._accumulator.accumulate(chunk)

                # Track finish reason
                choice = chunk.choices[0] if chunk.choices else None
                if choice and choice.finish_reason:
                    self._finish_reason = choice.finish_reason

                return chunk
            except StopAsyncIteration:
                # Stream ended — check if we need to handle tool calls
                if await self._handle_stream_end():
                    # New stream started, continue yielding
                    continue
                raise

    async def _handle_stream_end(self) -> bool:
        """Handle end of a stream. Returns True if a new stream was started."""
        tool_calls = self._accumulator.get_tool_calls()

        if (
            not tool_calls
            or not self._inject_tools
            or self._rounds >= self._max_rounds
            or self._finish_reason != "tool_calls"
        ):
            # Final stream — fire background ingest
            if self._auto_ingest and self._messages:
                asyncio.create_task(
                    background_ingest(
                        self._store, self._messages,
                        self._org_id, self._agent_id,
                    )
                )
            self._finished = True
            return False

        # Has memory tool calls — execute them
        memory_calls = [
            tc for tc in tool_calls
            if self._executor.is_memory_tool_call(tc["function"]["name"])
        ]
        if not memory_calls:
            if self._auto_ingest and self._messages:
                asyncio.create_task(
                    background_ingest(
                        self._store, self._messages,
                        self._org_id, self._agent_id,
                    )
                )
            self._finished = True
            return False

        # Add assistant message with tool calls to conversation
        self._messages.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for tc in tool_calls
            ],
        })

        # Execute memory tool calls
        tool_results = await self._executor.handle_openai_tool_calls(tool_calls)
        self._messages.extend(tool_results)

        # Start new stream
        self._accumulator.reset()
        self._finish_reason = None
        self._rounds += 1
        self._stream = await self._client.chat.completions.create(
            messages=self._messages, **self._kwargs
        )
        return True


class WrappedOpenAI:
    """Proxy for AsyncOpenAI that adds memory to chat completions."""

    def __init__(self, client: Any, memory_chat: MemoryChat):
        self._client = client
        self._memory_chat = memory_chat

    @property
    def chat(self):
        return self._memory_chat

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the original client."""
        return getattr(self._client, name)


def wrap_openai(
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
) -> WrappedOpenAI:
    """Wrap an AsyncOpenAI client with transparent memory.

    Args:
        client: An AsyncOpenAI() instance.
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
    memory_chat = MemoryChat(
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
    return WrappedOpenAI(client, memory_chat)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_last_user_message(messages: list[dict]) -> str | None:
    """Find the last user message content."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            # Handle list content (vision messages)
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
    return None


def _inject_system_context(messages: list[dict], context: str) -> list[dict]:
    """Inject context into the system message. Creates one if missing."""
    messages = list(messages)
    if messages and messages[0].get("role") == "system":
        messages[0] = {
            **messages[0],
            "content": messages[0].get("content", "") + context,
        }
    else:
        messages.insert(0, {"role": "system", "content": context})
    return messages
