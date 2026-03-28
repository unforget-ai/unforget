"""Tests for streaming support in OpenAI and Anthropic integrations."""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from unforget.integrations.anthropic import wrap_anthropic
from unforget.integrations.openai import wrap_openai
from unforget.integrations.streaming import (
    AnthropicToolAccumulator,
    OpenAIToolAccumulator,
)

ORG = "test-org"
AGENT = "test-agent"


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


@dataclass
class MockDelta:
    content: str | None = None
    tool_calls: list | None = None


@dataclass
class MockToolCallDelta:
    index: int = 0
    id: str | None = None
    function: Any = None


@dataclass
class MockFunctionDelta:
    name: str | None = None
    arguments: str | None = None


@dataclass
class MockChoice:
    delta: MockDelta | None = None
    finish_reason: str | None = None


@dataclass
class MockChunk:
    choices: list[MockChoice] = field(default_factory=list)


def _text_chunk(text: str, finish_reason: str | None = None) -> MockChunk:
    return MockChunk(choices=[MockChoice(
        delta=MockDelta(content=text),
        finish_reason=finish_reason,
    )])


def _tool_call_start_chunk(index: int, tool_id: str, name: str) -> MockChunk:
    return MockChunk(choices=[MockChoice(
        delta=MockDelta(tool_calls=[MockToolCallDelta(
            index=index,
            id=tool_id,
            function=MockFunctionDelta(name=name, arguments=""),
        )]),
    )])


def _tool_call_args_chunk(index: int, args_fragment: str) -> MockChunk:
    return MockChunk(choices=[MockChoice(
        delta=MockDelta(tool_calls=[MockToolCallDelta(
            index=index,
            function=MockFunctionDelta(arguments=args_fragment),
        )]),
    )])


def _tool_call_end_chunk() -> MockChunk:
    return MockChunk(choices=[MockChoice(
        delta=MockDelta(),
        finish_reason="tool_calls",
    )])


async def _async_iter(items):
    """Create an async iterator from a list."""
    for item in items:
        yield item


class MockStore:
    """Minimal MemoryStore mock for testing wrappers."""

    def __init__(self):
        self.auto_recall = AsyncMock(return_value=None)
        self.ingest = AsyncMock()
        self.write = AsyncMock(return_value=MagicMock(
            id="mem-1", content="stored"
        ))
        self.recall = AsyncMock(return_value=[])
        self.list = AsyncMock(return_value=[])


# ---------------------------------------------------------------------------
# OpenAI Tool Accumulator Tests
# ---------------------------------------------------------------------------


class TestOpenAIToolAccumulator:
    def test_empty_stream(self):
        acc = OpenAIToolAccumulator()
        assert acc.get_tool_calls() == []

    def test_text_only_chunks(self):
        acc = OpenAIToolAccumulator()
        acc.accumulate(_text_chunk("hello"))
        acc.accumulate(_text_chunk(" world"))
        assert acc.get_tool_calls() == []

    def test_merges_deltas(self):
        acc = OpenAIToolAccumulator()
        acc.accumulate(_tool_call_start_chunk(0, "call_1", "memory_store"))
        acc.accumulate(_tool_call_args_chunk(0, '{"conte'))
        acc.accumulate(_tool_call_args_chunk(0, 'nt": "hello"}'))

        calls = acc.get_tool_calls()
        assert len(calls) == 1
        assert calls[0]["id"] == "call_1"
        assert calls[0]["function"]["name"] == "memory_store"
        assert json.loads(calls[0]["function"]["arguments"]) == {"content": "hello"}

    def test_multiple_tool_calls(self):
        acc = OpenAIToolAccumulator()
        acc.accumulate(_tool_call_start_chunk(0, "call_1", "memory_store"))
        acc.accumulate(_tool_call_args_chunk(0, '{"content": "a"}'))
        acc.accumulate(_tool_call_start_chunk(1, "call_2", "memory_search"))
        acc.accumulate(_tool_call_args_chunk(1, '{"query": "b"}'))

        calls = acc.get_tool_calls()
        assert len(calls) == 2
        assert calls[0]["function"]["name"] == "memory_store"
        assert calls[1]["function"]["name"] == "memory_search"

    def test_reset(self):
        acc = OpenAIToolAccumulator()
        acc.accumulate(_tool_call_start_chunk(0, "call_1", "memory_store"))
        acc.reset()
        assert acc.get_tool_calls() == []


# ---------------------------------------------------------------------------
# Anthropic Tool Accumulator Tests
# ---------------------------------------------------------------------------


@dataclass
class MockAnthropicEvent:
    type: str
    index: int = 0
    content_block: Any = None
    delta: Any = None


@dataclass
class MockContentBlock:
    type: str
    id: str = ""
    name: str = ""


@dataclass
class MockInputDelta:
    type: str = "input_json_delta"
    partial_json: str = ""


class TestAnthropicToolAccumulator:
    def test_empty_stream(self):
        acc = AnthropicToolAccumulator()
        assert acc.get_tool_blocks() == []

    def test_merges_blocks(self):
        acc = AnthropicToolAccumulator()
        acc.accumulate(MockAnthropicEvent(
            type="content_block_start",
            index=0,
            content_block=MockContentBlock(type="tool_use", id="tu_1", name="memory_store"),
        ))
        acc.accumulate(MockAnthropicEvent(
            type="content_block_delta",
            index=0,
            delta=MockInputDelta(partial_json='{"conte'),
        ))
        acc.accumulate(MockAnthropicEvent(
            type="content_block_delta",
            index=0,
            delta=MockInputDelta(partial_json='nt": "hello"}'),
        ))

        blocks = acc.get_tool_blocks()
        assert len(blocks) == 1
        assert blocks[0]["id"] == "tu_1"
        assert blocks[0]["name"] == "memory_store"
        assert blocks[0]["input"] == {"content": "hello"}

    def test_multiple_blocks(self):
        acc = AnthropicToolAccumulator()
        acc.accumulate(MockAnthropicEvent(
            type="content_block_start",
            index=0,
            content_block=MockContentBlock(type="tool_use", id="tu_1", name="memory_store"),
        ))
        acc.accumulate(MockAnthropicEvent(
            type="content_block_delta",
            index=0,
            delta=MockInputDelta(partial_json='{"content": "a"}'),
        ))
        acc.accumulate(MockAnthropicEvent(
            type="content_block_start",
            index=1,
            content_block=MockContentBlock(type="tool_use", id="tu_2", name="memory_search"),
        ))
        acc.accumulate(MockAnthropicEvent(
            type="content_block_delta",
            index=1,
            delta=MockInputDelta(partial_json='{"query": "b"}'),
        ))

        blocks = acc.get_tool_blocks()
        assert len(blocks) == 2
        assert blocks[0]["name"] == "memory_store"
        assert blocks[1]["name"] == "memory_search"

    def test_reset(self):
        acc = AnthropicToolAccumulator()
        acc.accumulate(MockAnthropicEvent(
            type="content_block_start",
            index=0,
            content_block=MockContentBlock(type="tool_use", id="tu_1", name="memory_store"),
        ))
        acc.reset()
        assert acc.get_tool_blocks() == []


# ---------------------------------------------------------------------------
# OpenAI Streaming Integration Tests
# ---------------------------------------------------------------------------


class TestOpenAIStreaming:
    @pytest.fixture
    def mock_store(self):
        return MockStore()

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_stream_passthrough(self, mock_client, mock_store):
        """Text-only stream: all chunks yielded, no tool execution."""
        chunks = [
            _text_chunk("Hello"),
            _text_chunk(" world"),
            _text_chunk("!", finish_reason="stop"),
        ]
        mock_client.chat.completions.create.return_value = _async_iter(chunks)

        wrapped = wrap_openai(
            mock_client, mock_store, ORG, AGENT,
            inject_tools=False, auto_recall=False, auto_ingest=False,
        )
        result = []
        stream = await wrapped.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o", stream=True,
        )
        async for chunk in stream:
            c = chunk.choices[0].delta.content
            if c:
                result.append(c)

        assert result == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_stream_with_tool_call(self, mock_client, mock_store):
        """Stream with memory tool call: tools executed, second stream yielded."""
        # First stream: tool call
        first_stream = [
            _tool_call_start_chunk(0, "call_1", "memory_store"),
            _tool_call_args_chunk(0, '{"content": "user likes Python"}'),
            _tool_call_end_chunk(),
        ]
        # Second stream: text response
        second_stream = [
            _text_chunk("Got it!"),
            _text_chunk("", finish_reason="stop"),
        ]

        mock_client.chat.completions.create.side_effect = [
            _async_iter(first_stream),
            _async_iter(second_stream),
        ]

        wrapped = wrap_openai(
            mock_client, mock_store, ORG, AGENT,
            auto_recall=False, auto_ingest=False,
        )

        result = []
        stream = await wrapped.chat.completions.create(
            messages=[{"role": "user", "content": "I like Python"}],
            model="gpt-4o", stream=True,
        )
        async for chunk in stream:
            c = chunk.choices[0].delta.content
            if c:
                result.append(c)

        assert result == ["Got it!"]
        # Verify write was called (memory_store tool executed)
        mock_store.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_auto_recall_injected(self, mock_client, mock_store):
        """Auto-recall injects memory context into system prompt."""
        mock_store.auto_recall.return_value = "[Memory: user likes Python]"

        chunks = [_text_chunk("ok", finish_reason="stop")]
        mock_client.chat.completions.create.return_value = _async_iter(chunks)

        wrapped = wrap_openai(
            mock_client, mock_store, ORG, AGENT,
            inject_tools=False, auto_ingest=False,
        )

        stream = await wrapped.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o", stream=True,
        )
        async for _ in stream:
            pass

        # Check the messages passed to OpenAI include memory context
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert any("[Memory: user likes Python]" in str(m.get("content", "")) for m in messages)

    @pytest.mark.asyncio
    async def test_stream_auto_ingest(self, mock_client, mock_store):
        """Auto-ingest fires after stream ends."""
        chunks = [_text_chunk("hi", finish_reason="stop")]
        mock_client.chat.completions.create.return_value = _async_iter(chunks)

        wrapped = wrap_openai(
            mock_client, mock_store, ORG, AGENT,
            inject_tools=False, auto_recall=False,
        )

        stream = await wrapped.chat.completions.create(
            messages=[{"role": "user", "content": "hello"}],
            model="gpt-4o", stream=True,
        )
        async for _ in stream:
            pass

        # Give background task a moment to run
        await asyncio.sleep(0.05)
        mock_store.ingest.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_multi_round(self, mock_client, mock_store):
        """Two tool call rounds before final content."""
        # Round 1: memory_store
        stream1 = [
            _tool_call_start_chunk(0, "call_1", "memory_store"),
            _tool_call_args_chunk(0, '{"content": "fact 1"}'),
            _tool_call_end_chunk(),
        ]
        # Round 2: memory_store again
        stream2 = [
            _tool_call_start_chunk(0, "call_2", "memory_store"),
            _tool_call_args_chunk(0, '{"content": "fact 2"}'),
            _tool_call_end_chunk(),
        ]
        # Final: text
        stream3 = [
            _text_chunk("Done!"),
            _text_chunk("", finish_reason="stop"),
        ]

        mock_client.chat.completions.create.side_effect = [
            _async_iter(stream1),
            _async_iter(stream2),
            _async_iter(stream3),
        ]

        wrapped = wrap_openai(
            mock_client, mock_store, ORG, AGENT,
            auto_recall=False, auto_ingest=False,
        )

        result = []
        stream = await wrapped.chat.completions.create(
            messages=[{"role": "user", "content": "remember two things"}],
            model="gpt-4o", stream=True,
        )
        async for chunk in stream:
            c = chunk.choices[0].delta.content
            if c:
                result.append(c)

        assert result == ["Done!"]
        assert mock_store.write.call_count == 2


# ---------------------------------------------------------------------------
# Anthropic Streaming Integration Tests
# ---------------------------------------------------------------------------


@dataclass
class MockAnthropicMessageDelta:
    type: str = "message_delta"
    delta: Any = None


@dataclass
class MockStopDelta:
    stop_reason: str = "end_turn"


@dataclass
class MockToolStopDelta:
    stop_reason: str = "tool_use"


def _anthropic_text_event(text: str) -> MockAnthropicEvent:
    return MockAnthropicEvent(type="content_block_delta", delta=MagicMock(
        type="text_delta", text=text,
    ))


def _anthropic_message_stop(stop_reason: str = "end_turn") -> MockAnthropicMessageDelta:
    delta = MagicMock()
    delta.stop_reason = stop_reason
    return MockAnthropicMessageDelta(type="message_delta", delta=delta)


class TestAnthropicStreaming:
    @pytest.fixture
    def mock_store(self):
        return MockStore()

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.messages = MagicMock()
        client.messages.create = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_stream_passthrough(self, mock_client, mock_store):
        """Text-only stream: all events yielded."""
        events = [
            MockAnthropicEvent(type="message_start"),
            _anthropic_text_event("Hello"),
            _anthropic_message_stop("end_turn"),
            MockAnthropicEvent(type="message_stop"),
        ]
        mock_client.messages.create.return_value = _async_iter(events)

        wrapped = wrap_anthropic(
            mock_client, mock_store, ORG, AGENT,
            inject_tools=False, auto_recall=False, auto_ingest=False,
        )

        collected = []
        stream = await wrapped.messages.create(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-6", max_tokens=100, stream=True,
        )
        async for event in stream:
            collected.append(event)

        # All events should be yielded
        assert len(collected) == 4

    @pytest.mark.asyncio
    async def test_stream_with_tool_call(self, mock_client, mock_store):
        """Stream with memory tool call: tools executed, second stream yielded."""
        # First stream: tool_use
        first_events = [
            MockAnthropicEvent(type="message_start"),
            MockAnthropicEvent(
                type="content_block_start", index=0,
                content_block=MockContentBlock(type="tool_use", id="tu_1", name="memory_store"),
            ),
            MockAnthropicEvent(
                type="content_block_delta", index=0,
                delta=MockInputDelta(partial_json='{"content": "user likes Go"}'),
            ),
            MockAnthropicEvent(type="content_block_stop", index=0),
            _anthropic_message_stop("tool_use"),
            MockAnthropicEvent(type="message_stop"),
        ]
        # Second stream: text
        second_events = [
            MockAnthropicEvent(type="message_start"),
            _anthropic_text_event("Got it!"),
            _anthropic_message_stop("end_turn"),
            MockAnthropicEvent(type="message_stop"),
        ]

        mock_client.messages.create.side_effect = [
            _async_iter(first_events),
            _async_iter(second_events),
        ]

        wrapped = wrap_anthropic(
            mock_client, mock_store, ORG, AGENT,
            auto_recall=False, auto_ingest=False,
        )

        collected = []
        stream = await wrapped.messages.create(
            messages=[{"role": "user", "content": "I like Go"}],
            model="claude-sonnet-4-6", max_tokens=100, stream=True,
        )
        async for event in stream:
            collected.append(event)

        # Should have events from both streams
        assert len(collected) > 4
        mock_store.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_auto_recall_injected(self, mock_client, mock_store):
        """Auto-recall injects memory into system kwarg."""
        mock_store.auto_recall.return_value = "[Memory: user likes Go]"

        events = [
            MockAnthropicEvent(type="message_start"),
            _anthropic_message_stop("end_turn"),
            MockAnthropicEvent(type="message_stop"),
        ]
        mock_client.messages.create.return_value = _async_iter(events)

        wrapped = wrap_anthropic(
            mock_client, mock_store, ORG, AGENT,
            inject_tools=False, auto_ingest=False,
        )

        stream = await wrapped.messages.create(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-6", max_tokens=100, stream=True,
        )
        async for _ in stream:
            pass

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "[Memory: user likes Go]" in call_kwargs.get("system", "")

    @pytest.mark.asyncio
    async def test_stream_auto_ingest(self, mock_client, mock_store):
        """Auto-ingest fires after stream ends."""
        events = [
            MockAnthropicEvent(type="message_start"),
            _anthropic_message_stop("end_turn"),
            MockAnthropicEvent(type="message_stop"),
        ]
        mock_client.messages.create.return_value = _async_iter(events)

        wrapped = wrap_anthropic(
            mock_client, mock_store, ORG, AGENT,
            inject_tools=False, auto_recall=False,
        )

        stream = await wrapped.messages.create(
            messages=[{"role": "user", "content": "hello"}],
            model="claude-sonnet-4-6", max_tokens=100, stream=True,
        )
        async for _ in stream:
            pass

        await asyncio.sleep(0.05)
        mock_store.ingest.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_multi_round(self, mock_client, mock_store):
        """Two tool call rounds before final content."""
        def _tool_stream(tool_id, name, args_json):
            return [
                MockAnthropicEvent(type="message_start"),
                MockAnthropicEvent(
                    type="content_block_start", index=0,
                    content_block=MockContentBlock(type="tool_use", id=tool_id, name=name),
                ),
                MockAnthropicEvent(
                    type="content_block_delta", index=0,
                    delta=MockInputDelta(partial_json=args_json),
                ),
                MockAnthropicEvent(type="content_block_stop", index=0),
                _anthropic_message_stop("tool_use"),
                MockAnthropicEvent(type="message_stop"),
            ]

        final_events = [
            MockAnthropicEvent(type="message_start"),
            _anthropic_text_event("All done!"),
            _anthropic_message_stop("end_turn"),
            MockAnthropicEvent(type="message_stop"),
        ]

        mock_client.messages.create.side_effect = [
            _async_iter(_tool_stream("tu_1", "memory_store", '{"content": "fact 1"}')),
            _async_iter(_tool_stream("tu_2", "memory_store", '{"content": "fact 2"}')),
            _async_iter(final_events),
        ]

        wrapped = wrap_anthropic(
            mock_client, mock_store, ORG, AGENT,
            auto_recall=False, auto_ingest=False,
        )

        collected = []
        stream = await wrapped.messages.create(
            messages=[{"role": "user", "content": "remember two things"}],
            model="claude-sonnet-4-6", max_tokens=100, stream=True,
        )
        async for event in stream:
            collected.append(event)

        assert mock_store.write.call_count == 2
