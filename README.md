<p align="center">
  <a href="https://unforget.sh">
    <img src="https://docs.unforget.sh/logo.png" width="180" alt="Unforget">
  </a>
</p>

<h1 align="center">Unforget</h1>

<p align="center">
  <strong>Zero-LLM memory for AI agents. One database. Nothing to forget.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/unforget"><img src="https://img.shields.io/pypi/v/unforget?color=%2334D058&label=pypi" alt="PyPI"></a>
  <a href="https://pypi.org/project/unforget"><img src="https://img.shields.io/pypi/pyversions/unforget" alt="Python"></a>
  <a href="https://github.com/unforget-ai/unforget/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  <a href="https://docs.unforget.sh"><img src="https://img.shields.io/badge/docs-unforget.sh-blue" alt="Docs"></a>
</p>

<p align="center">
  <a href="https://docs.unforget.sh">Documentation</a> · <a href="#quick-start">Quick Start</a> · <a href="#llm-integrations">Integrations</a> · <a href="#mcp-server">MCP</a> · <a href="#how-retrieval-works">How It Works</a>
</p>

---

Most agent memory solutions require an LLM call on every write, adding 500ms+ latency and API costs. Unforget stores memories in **~7ms** with zero LLM calls, retrieves them with **4-channel hybrid search**, and runs on a single PostgreSQL database you already know how to operate.

```bash
pip install unforget
```

### Why Unforget?

| | Unforget | Others (Mem0, Zep, etc.) |
|---|---|---|
| **Write latency** | ~7ms | 500ms+ (LLM required) |
| **Write cost** | $0 | LLM API cost per write |
| **Retrieval** | 4-channel hybrid (semantic + BM25 + entity + temporal) | Vector-only or vector + graph |
| **Infrastructure** | PostgreSQL only | PostgreSQL + Neo4j / Qdrant / etc. |
| **LLM dependency** | None on write path | Required on every operation |

---

## Quick Start

```python
from unforget import MemoryStore

store = MemoryStore("postgresql://user:pass@localhost/db")
await store.initialize()

# Bind once — no more repeating org_id and agent_id
memory = store.bind(org_id="acme", agent_id="support-bot")

# Write — instant, no LLM calls
await memory.write("User prefers dark mode")
await memory.write("Last order was #4821, shipped March 20",
    memory_type="event", tags=["orders"], importance=0.8)

# Recall — 4-channel retrieval + cross-encoder reranking
results = await memory.recall("what did the user order?")
# → [MemoryResult(content="Last order was #4821, shipped March 20", score=0.94)]

# Auto-recall — formatted context ready to inject into a system prompt
context = await memory.auto_recall("help with their order")
# → "[Memory Context]\n- Last order was #4821, shipped March 20\n- User prefers dark mode"
```

---

## LLM Integrations

Wrap your existing OpenAI or Anthropic client. Memory becomes transparent — your agent gets recall, tools, and ingestion without changing your code.

### OpenAI

```bash
pip install unforget[openai]
```

```python
from openai import AsyncOpenAI
from unforget import MemoryStore
from unforget.integrations.openai import wrap_openai

store = MemoryStore("postgresql://user:pass@localhost/db")
await store.initialize()

client = wrap_openai(AsyncOpenAI(), store, org_id="acme", agent_id="support-bot")

# That's it. Memory is handled automatically:
# - Relevant memories injected into system prompt
# - Agent can call memory_store, memory_search, etc.
# - Conversation saved after each response
response = await client.chat.completions.create(
    messages=[{"role": "user", "content": "What was my last order?"}],
    model="gpt-4o",
)
```

### Anthropic

```bash
pip install unforget[anthropic]
```

```python
from anthropic import AsyncAnthropic
from unforget.integrations.anthropic import wrap_anthropic

client = wrap_anthropic(AsyncAnthropic(), store, org_id="acme", agent_id="support-bot")

response = await client.messages.create(
    messages=[{"role": "user", "content": "What was my last order?"}],
    model="claude-sonnet-4-6",
    max_tokens=1024,
)
```

### What the wrappers do

1. **Auto-recall** — searches memory before each LLM call, injects relevant context
2. **Memory tools** — 5 tools the agent can call: `memory_store`, `memory_search`, `memory_list`, `memory_update`, `memory_forget`
3. **Tool execution** — handles the tool call loop automatically (up to 5 rounds)
4. **Auto-ingest** — stores the conversation in background after the response

Everything is configurable:

```python
client = wrap_openai(
    AsyncOpenAI(), store, org_id="acme", agent_id="support-bot",
    auto_recall=True,         # inject memory context into system prompt
    auto_ingest=True,         # store conversation after response
    inject_tools=True,        # add memory tools to every call
    inject_instructions=True, # add memory usage instructions
    tools=["memory_store", "memory_search"],  # pick which tools to enable
)
```

### Framework-Agnostic

Use `MemoryToolExecutor` directly with LangChain, CrewAI, or any framework:

```python
from unforget import MemoryToolExecutor

executor = MemoryToolExecutor(store, org_id="acme", agent_id="support-bot")

# Get tool schemas in your provider's format
tools = executor.to_openai()      # OpenAI format
tools = executor.to_anthropic()   # Anthropic format
tools = executor.to_generic()     # Raw dict format

# Execute a tool call from any LLM response
result = await executor.execute("memory_store", {
    "content": "User's favorite color is blue",
    "memory_type": "insight",
    "importance": 0.7,
})
```

---

## MCP Server

Use Unforget with Claude Code, Cursor, or any MCP client:

```bash
pipx install unforget-mcp
```

Add to `.mcp.json`:

```json
{
  "mcpServers": {
    "unforget": {
      "command": "unforget-mcp",
      "args": []
    }
  }
}
```

5 tools: `memory_store`, `memory_search`, `memory_forget`, `memory_list`, `memory_update`. Zero config — embedded PostgreSQL starts automatically.

See [unforget-mcp](https://github.com/unforget-ai/unforget-mcp) for details.

---

## How Retrieval Works

Every `recall()` fires four search channels in parallel inside a single SQL query, then fuses results with Reciprocal Rank Fusion:

| Channel | What it does | Index |
|---------|-------------|-------|
| **Semantic** | pgvector cosine similarity | HNSW |
| **BM25** | PostgreSQL full-text search | GIN (tsvector) |
| **Entity** | Named entity overlap | GIN (array) |
| **Temporal** | Recently accessed memories first | B-tree |

Results are boosted by type (insight > event > raw) and reranked with a cross-encoder for accuracy.

**One SQL round trip. Four search strategies. No external services.**

---

## Benchmark: LoCoMo (ACL 2024)

Tested on [LoCoMo](https://github.com/snap-research/locomo) — 1,540 questions across 10 long conversations measuring memory accuracy across single-hop, multi-hop, temporal, and open-domain categories.

| Category | Accuracy |
|----------|----------|
| Single-hop | 73.0% |
| Temporal | 81.0% |
| Multi-hop | 66.7% |
| Open-domain | 85.0% |
| **Overall** | **80.6%** |

Config: `--ingest-mode hybrid --recall-limit 15`, answer/judge model: `claude-haiku-4-5`. Zero-LLM write path — retrieval only, no RAG chain.

---

## More Examples

<details>
<summary><strong>Personal assistant that remembers preferences</strong></summary>

```python
memory = store.bind(org_id="user-123", agent_id="scheduler")

# First conversation
await memory.write("Likes morning meetings, never after 3pm")
await memory.write("Allergic to shellfish")
await memory.write("Prefers window seats on flights")

# Weeks later...
context = await memory.auto_recall("book a dinner reservation")
# → recalls the shellfish allergy automatically
```
</details>

<details>
<summary><strong>Customer support bot with history</strong></summary>

```python
support = store.bind(org_id="acme", agent_id="support")

# Ingest a full conversation transcript
await support.ingest([
    {"role": "user", "content": "My printer isn't connecting to wifi"},
    {"role": "assistant", "content": "Let's try resetting the network settings..."},
    {"role": "user", "content": "That worked! Thanks."},
], mode="background")

# Next time the user calls about printers:
results = await support.recall("printer wifi issue")
# → recalls the previous fix
```
</details>

<details>
<summary><strong>Fact versioning — when things change</strong></summary>

```python
memory = store.bind(org_id="u-1", agent_id="bot")

# User moves to a new city
m = await memory.write("User lives in Austin, TX")

# Six months later, they move
old, new = await memory.supersede(m.id, "User lives in Denver, CO")
# Old memory soft-deleted, new one linked. Full audit trail.

# "Where did they live in January?"
memories = await memory.timeline(at=january_15)
# → [MemoryItem(content="User lives in Austin, TX")]
```
</details>

<details>
<summary><strong>Background consolidation</strong></summary>

```python
from unforget import ConsolidationScheduler

# Run consolidation every hour
scheduler = ConsolidationScheduler(store, interval_seconds=3600)
store.attach_scheduler(scheduler)
await scheduler.start()

# Or trigger manually
memory = store.bind(org_id="acme", agent_id="bot")
report = await memory.consolidate()
# → ConsolidationReport(duplicates_merged=3, memories_decayed=12, memories_expired=5)
```

Consolidation handles:
- **Dedup** — merges near-identical memories (cosine > 0.92)
- **Decay** — reduces importance of memories not accessed in 7/30 days
- **Expire** — soft-deletes raw chunks past their 30-day TTL
- **Promote** — distills raw conversation chunks into insights (with optional LLM)
</details>

<details>
<summary><strong>REST API — 17 endpoints out of the box</strong></summary>

```python
from unforget.api import create_memory_router

app.include_router(create_memory_router(store), prefix="/v1/memory")
# write, recall, auto-recall, ingest, list, get, update, delete,
# bulk-delete, supersede, timeline, chain, history, stats, consolidate
```
</details>

<details>
<summary><strong>Pluggable embedders</strong></summary>

```python
from unforget import MemoryStore, OpenAIEmbedder

# Default: local ONNX/PyTorch (free, ~3ms per embed)
store = MemoryStore("postgresql://...")

# OpenAI: higher quality, costs money
store = MemoryStore("postgresql://...", embedder=OpenAIEmbedder())

# Custom: implement BaseEmbedder
from unforget import BaseEmbedder

class CohereEmbedder(BaseEmbedder):
    @property
    def dims(self) -> int:
        return 1024

    def embed(self, text: str) -> list[float]:
        return cohere_client.embed([text]).embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return cohere_client.embed(texts).embeddings

store = MemoryStore("postgresql://...", embedder=CohereEmbedder())
```
</details>

---

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| `write()` | ~7ms | ONNX embed + single SQL insert |
| `recall()` | ~25ms | Embed + 4-channel CTE + rerank |
| `auto_recall()` | ~25ms | recall + format for system prompt |
| `write_batch(20)` | ~65ms | Batch embedding + batch insert |
| Cache hit (recall) | <0.1ms | TTL cache for repeated queries |
| Cache hit (embed) | <0.1ms | LRU cache by content hash |

---

## Infrastructure

```bash
# Start PostgreSQL + pgvector
docker compose up -d

# Install
pip install unforget              # core only
pip install unforget[openai]      # + OpenAI SDK wrapper
pip install unforget[anthropic]   # + Anthropic SDK wrapper
pip install unforget[api]         # + FastAPI router
pip install unforget[spacy]       # + better entity extraction
```

- **Database**: PostgreSQL 16 + pgvector
- **Embedding**: `all-MiniLM-L6-v2` via ONNX Runtime (default) or PyTorch
- **Reranking**: `ms-marco-MiniLM-L-6-v2` cross-encoder
- **Python**: 3.11+
- **No external APIs** required for core functionality

---

## License

Apache 2.0
