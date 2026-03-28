"""Tests for memory tools and LLM integrations — requires PostgreSQL + pgvector."""

import json
from unittest.mock import AsyncMock, MagicMock

from unforget import MEMORY_TOOLS, MemoryToolExecutor, MemoryType
from unforget.tools import MEMORY_SYSTEM_INSTRUCTION

from .conftest import requires_db

ORG = "test-org"
AGENT = "test-agent"


class TestToolSchemas:
    """Unit tests — no database needed."""

    def test_all_tools_defined(self):
        names = {t["name"] for t in MEMORY_TOOLS}
        assert names == {"memory_store", "memory_search", "memory_list", "memory_forget", "memory_update"}

    def test_all_have_parameters(self):
        for tool in MEMORY_TOOLS:
            assert "parameters" in tool
            assert tool["parameters"]["type"] == "object"

    def test_required_fields(self):
        store_tool = next(t for t in MEMORY_TOOLS if t["name"] == "memory_store")
        assert "content" in store_tool["parameters"]["required"]

        search_tool = next(t for t in MEMORY_TOOLS if t["name"] == "memory_search")
        assert "query" in search_tool["parameters"]["required"]

    def test_system_instruction_exists(self):
        assert "[Memory Instructions]" in MEMORY_SYSTEM_INSTRUCTION
        assert "memory_store" in MEMORY_SYSTEM_INSTRUCTION


@requires_db
class TestToolExecutor:
    async def test_to_openai_format(self, store):
        executor = MemoryToolExecutor(store, ORG, AGENT)
        tools = executor.to_openai()
        assert len(tools) == 5
        for t in tools:
            assert t["type"] == "function"
            assert "function" in t
            assert "name" in t["function"]
            assert "parameters" in t["function"]

    async def test_to_anthropic_format(self, store):
        executor = MemoryToolExecutor(store, ORG, AGENT)
        tools = executor.to_anthropic()
        assert len(tools) == 5
        for t in tools:
            assert "name" in t
            assert "input_schema" in t

    async def test_subset_tools(self, store):
        executor = MemoryToolExecutor(
            store, ORG, AGENT, tools=["memory_store", "memory_search"]
        )
        assert len(executor.to_openai()) == 2
        assert len(executor.to_anthropic()) == 2

    async def test_exec_store(self, store):
        executor = MemoryToolExecutor(store, ORG, AGENT)
        result = await executor.execute("memory_store", {
            "content": "User prefers Python",
        })
        assert "Stored" in result
        assert "Python" in result

        # Verify it was actually stored
        items = await store.list(org_id=ORG, agent_id=AGENT)
        assert len(items) == 1
        assert items[0].content == "User prefers Python"

    async def test_exec_store_with_type_and_tags(self, store):
        executor = MemoryToolExecutor(store, ORG, AGENT)
        await executor.execute("memory_store", {
            "content": "Deployed on March 15",
            "memory_type": "event",
            "tags": ["deploy"],
        })
        items = await store.list(org_id=ORG, agent_id=AGENT)
        assert items[0].memory_type == MemoryType.EVENT
        assert items[0].tags == ["deploy"]

    async def test_exec_search(self, store):
        executor = MemoryToolExecutor(store, ORG, AGENT)
        await store.write("User deploys to Fly.io", org_id=ORG, agent_id=AGENT)

        result = await executor.execute("memory_search", {"query": "deployment"})
        assert "Found memories" in result
        assert "Fly.io" in result

    async def test_exec_search_empty(self, store):
        executor = MemoryToolExecutor(store, ORG, AGENT)
        result = await executor.execute("memory_search", {"query": "nothing"})
        assert "No memories found" in result

    async def test_exec_list(self, store):
        executor = MemoryToolExecutor(store, ORG, AGENT)
        await store.write("fact one", org_id=ORG, agent_id=AGENT)
        await store.write("fact two", org_id=ORG, agent_id=AGENT)

        result = await executor.execute("memory_list", {})
        assert "2 memories" in result

    async def test_exec_forget(self, store):
        executor = MemoryToolExecutor(store, ORG, AGENT)
        item = await store.write("forget me", org_id=ORG, agent_id=AGENT)

        result = await executor.execute("memory_forget", {"memory_id": str(item.id)})
        assert "deleted" in result.lower()

        items = await store.list(org_id=ORG, agent_id=AGENT)
        assert len(items) == 0

    async def test_exec_forget_invalid_id(self, store):
        executor = MemoryToolExecutor(store, ORG, AGENT)
        result = await executor.execute("memory_forget", {"memory_id": "not-a-uuid"})
        assert "Error" in result

    async def test_exec_update(self, store):
        executor = MemoryToolExecutor(store, ORG, AGENT)
        item = await store.write("deploys to AWS", org_id=ORG, agent_id=AGENT)

        result = await executor.execute("memory_update", {
            "memory_id": str(item.id),
            "new_content": "deploys to Fly.io",
        })
        assert "Updated" in result
        assert "Fly.io" in result

    async def test_exec_unknown_tool(self, store):
        executor = MemoryToolExecutor(store, ORG, AGENT)
        result = await executor.execute("nonexistent_tool", {})
        assert "unknown tool" in result.lower()

    async def test_is_memory_tool_call(self, store):
        executor = MemoryToolExecutor(store, ORG, AGENT)
        assert executor.is_memory_tool_call("memory_store") is True
        assert executor.is_memory_tool_call("memory_search") is True
        assert executor.is_memory_tool_call("some_other_tool") is False


@requires_db
class TestOpenAIResponseHandler:
    async def test_handle_openai_response(self, store):
        """Mock an OpenAI response with tool calls."""
        executor = MemoryToolExecutor(store, ORG, AGENT)

        # Build a mock OpenAI response
        tool_call = MagicMock()
        tool_call.id = "call_123"
        tool_call.function.name = "memory_store"
        tool_call.function.arguments = json.dumps({"content": "User likes Python"})

        choice = MagicMock()
        choice.message.tool_calls = [tool_call]

        response = MagicMock()
        response.choices = [choice]

        results = await executor.handle_openai_response(response)
        assert len(results) == 1
        assert results[0]["role"] == "tool"
        assert results[0]["tool_call_id"] == "call_123"
        assert "Stored" in results[0]["content"]

    async def test_handle_openai_no_tool_calls(self, store):
        executor = MemoryToolExecutor(store, ORG, AGENT)

        choice = MagicMock()
        choice.message.tool_calls = None

        response = MagicMock()
        response.choices = [choice]

        results = await executor.handle_openai_response(response)
        assert results == []

    async def test_handle_openai_skips_non_memory_tools(self, store):
        executor = MemoryToolExecutor(store, ORG, AGENT)

        tool_call = MagicMock()
        tool_call.id = "call_456"
        tool_call.function.name = "web_search"  # not a memory tool
        tool_call.function.arguments = "{}"

        choice = MagicMock()
        choice.message.tool_calls = [tool_call]

        response = MagicMock()
        response.choices = [choice]

        results = await executor.handle_openai_response(response)
        assert results == []


@requires_db
class TestAnthropicResponseHandler:
    async def test_handle_anthropic_response(self, store):
        executor = MemoryToolExecutor(store, ORG, AGENT)

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_123"
        tool_block.name = "memory_store"
        tool_block.input = {"content": "User likes Go"}

        text_block = MagicMock()
        text_block.type = "text"

        response = MagicMock()
        response.content = [text_block, tool_block]

        results = await executor.handle_anthropic_response(response)
        assert len(results) == 1
        assert results[0]["type"] == "tool_result"
        assert results[0]["tool_use_id"] == "toolu_123"
        assert "Stored" in results[0]["content"]

    async def test_handle_anthropic_no_tool_use(self, store):
        executor = MemoryToolExecutor(store, ORG, AGENT)

        text_block = MagicMock()
        text_block.type = "text"

        response = MagicMock()
        response.content = [text_block]

        results = await executor.handle_anthropic_response(response)
        assert results == []


@requires_db
class TestOpenAIWrapper:
    async def test_wrap_openai_injects_tools(self, store):
        """Wrapper should add memory tools to the call."""
        from unforget.integrations.openai import wrap_openai

        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = None
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        wrapped = wrap_openai(mock_client, store, ORG, AGENT)

        await wrapped.chat.completions.create(
            messages=[{"role": "user", "content": "hello"}],
            model="gpt-4o",
        )

        # Verify tools were injected
        call_kwargs = mock_client.chat.completions.create.call_args
        tools = call_kwargs.kwargs.get("tools", [])
        tool_names = {t["function"]["name"] for t in tools}
        assert "memory_store" in tool_names
        assert "memory_search" in tool_names

    async def test_wrap_openai_injects_system_context(self, store):
        """Wrapper should inject memory context into system prompt."""
        from unforget.integrations.openai import wrap_openai

        # Pre-populate a memory
        await store.write("User prefers Fly.io", org_id=ORG, agent_id=AGENT)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = None
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        wrapped = wrap_openai(mock_client, store, ORG, AGENT)

        await wrapped.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helper."},
                {"role": "user", "content": "Where do we deploy?"},
            ],
            model="gpt-4o",
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages", [])
        system_msg = messages[0]["content"]
        assert "[Memory Context]" in system_msg
        assert "Fly.io" in system_msg

    async def test_wrap_openai_no_tools_option(self, store):
        """Can disable tool injection."""
        from unforget.integrations.openai import wrap_openai

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = None
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        wrapped = wrap_openai(
            mock_client, store, ORG, AGENT,
            inject_tools=False, auto_recall=False, inject_instructions=False,
        )

        await wrapped.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o",
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        tools = call_kwargs.kwargs.get("tools", [])
        assert len(tools) == 0


@requires_db
class TestAnthropicWrapper:
    async def test_wrap_anthropic_injects_tools(self, store):
        from unforget.integrations.anthropic import wrap_anthropic

        mock_client = MagicMock()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        mock_response.content = [text_block]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        wrapped = wrap_anthropic(mock_client, store, ORG, AGENT)

        await wrapped.messages.create(
            messages=[{"role": "user", "content": "hello"}],
            model="claude-sonnet-4-6",
            max_tokens=1024,
        )

        call_kwargs = mock_client.messages.create.call_args
        tools = call_kwargs.kwargs.get("tools", [])
        tool_names = {t["name"] for t in tools}
        assert "memory_store" in tool_names

    async def test_wrap_anthropic_injects_system(self, store):
        from unforget.integrations.anthropic import wrap_anthropic

        await store.write("Team uses Go", org_id=ORG, agent_id=AGENT)

        mock_client = MagicMock()
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        mock_response.content = [text_block]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        wrapped = wrap_anthropic(mock_client, store, ORG, AGENT)

        await wrapped.messages.create(
            messages=[{"role": "user", "content": "What language do we use?"}],
            model="claude-sonnet-4-6",
            max_tokens=1024,
        )

        call_kwargs = mock_client.messages.create.call_args
        system = call_kwargs.kwargs.get("system", "")
        assert "[Memory Context]" in system
        assert "Go" in system
