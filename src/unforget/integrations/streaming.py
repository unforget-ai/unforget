"""Shared streaming utilities for OpenAI and Anthropic integrations.

Accumulators buffer incremental tool call deltas from streaming responses
into complete tool call objects that can be executed.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from unforget.store import MemoryStore

logger = logging.getLogger("unforget.integrations.streaming")


class OpenAIToolAccumulator:
    """Buffers OpenAI streaming tool_call deltas into complete tool calls.

    OpenAI streams tool calls as incremental chunks with:
    - delta.tool_calls[i].index — which tool call this fragment belongs to
    - delta.tool_calls[i].id — set on the first chunk only
    - delta.tool_calls[i].function.name — set on the first chunk only
    - delta.tool_calls[i].function.arguments — appended incrementally
    """

    def __init__(self) -> None:
        self._tool_calls: dict[int, dict[str, Any]] = {}

    def accumulate(self, chunk: Any) -> None:
        """Process a ChatCompletionChunk — buffer any tool_call deltas."""
        choice = chunk.choices[0] if chunk.choices else None
        if not choice or not choice.delta:
            return

        tool_calls = getattr(choice.delta, "tool_calls", None)
        if not tool_calls:
            return

        for tc_delta in tool_calls:
            idx = tc_delta.index
            if idx not in self._tool_calls:
                self._tool_calls[idx] = {
                    "id": "",
                    "function": {"name": "", "arguments": ""},
                }

            entry = self._tool_calls[idx]
            if tc_delta.id:
                entry["id"] = tc_delta.id
            if tc_delta.function:
                if tc_delta.function.name:
                    entry["function"]["name"] = tc_delta.function.name
                if tc_delta.function.arguments:
                    entry["function"]["arguments"] += tc_delta.function.arguments

    def get_tool_calls(self) -> list[dict[str, Any]]:
        """Return completed tool calls sorted by index."""
        if not self._tool_calls:
            return []
        return [self._tool_calls[i] for i in sorted(self._tool_calls)]

    def reset(self) -> None:
        self._tool_calls.clear()


class AnthropicToolAccumulator:
    """Buffers Anthropic streaming events into complete tool_use blocks.

    Anthropic streams tool_use as events:
    - content_block_start with type=tool_use → sets id, name, starts input
    - content_block_delta with type=input_json_delta → appends partial_json
    - content_block_stop → block is complete
    """

    def __init__(self) -> None:
        self._blocks: dict[int, dict[str, Any]] = {}
        self._current_index: int = -1

    def accumulate(self, event: Any) -> None:
        """Process a streaming event."""
        event_type = getattr(event, "type", None)

        if event_type == "content_block_start":
            block = getattr(event, "content_block", None)
            if block and getattr(block, "type", None) == "tool_use":
                idx = getattr(event, "index", self._current_index + 1)
                self._current_index = idx
                self._blocks[idx] = {
                    "type": "tool_use",
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input_json": "",
                }

        elif event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta and getattr(delta, "type", None) == "input_json_delta":
                idx = getattr(event, "index", self._current_index)
                if idx in self._blocks:
                    self._blocks[idx]["input_json"] += getattr(
                        delta, "partial_json", ""
                    )

    def get_tool_blocks(self) -> list[dict[str, Any]]:
        """Return completed tool_use blocks with parsed input."""
        result = []
        for idx in sorted(self._blocks):
            block = self._blocks[idx]
            try:
                input_data = json.loads(block["input_json"]) if block["input_json"] else {}
            except json.JSONDecodeError:
                input_data = {}
            result.append({
                "type": "tool_use",
                "id": block["id"],
                "name": block["name"],
                "input": input_data,
            })
        return result

    def reset(self) -> None:
        self._blocks.clear()
        self._current_index = -1


async def background_ingest(
    store: MemoryStore,
    messages: list[dict],
    org_id: str,
    agent_id: str,
) -> None:
    """Fire-and-forget conversation ingest."""
    try:
        await store.ingest(
            messages,
            org_id=org_id,
            agent_id=agent_id,
            mode="background",
        )
    except Exception as e:
        logger.debug("Auto-ingest failed (non-fatal): %s", e)
