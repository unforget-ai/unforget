# Unforget

Memory for AI agents that actually works. PostgreSQL + pgvector. No vendor lock-in. No cloud dependency. Just fast, reliable memory your agents can read and write in milliseconds.

```bash
pip install unforget
```

---

## Why unforget?

Most agent memory solutions are either too slow (LLM on every write), too complex (graph databases, vector-only), or too locked-in (cloud APIs, proprietary formats).

Unforget is different:
- **Zero LLM on write** — memories are stored in ~7ms, not seconds
- **4-channel hybrid retrieval** — semantic + full-text + entity + temporal, fused with RRF
- **One database** — PostgreSQL with pgvector. You already know how to run it
- **ONNX Runtime** — 2-3x faster inference than PyTorch on CPU
- **Background consolidation** — dedup, decay, and promotion happen while your agent sleeps

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

## How Retrieval Works

Every `recall()` fires four search channels in parallel inside a single SQL query, then fuses results with Reciprocal Rank Fusion:

| Channel | What it does | Index |
|---------|-------------|-------|
| **Semantic** | pgvector cosine similarity | HNSW |
| **BM25** | PostgreSQL full-text search | GIN (tsvector) |
| **Entity** | Named entity overlap | GIN (array) |
| **Temporal** | Recently accessed memories first | B-tree |

Results are boosted by type (insight > event > raw) and reranked with a cross-encoder for accuracy.

One SQL round trip. Four search strategies. No external services.

---

## Pluggable Embedders

Swap the embedding model without changing your code. The default uses a local ONNX model (fast, free). Need better quality? Plug in OpenAI, Cohere, or your own.

```python
from unforget import MemoryStore, OpenAIEmbedder

# Default: local ONNX/PyTorch (free, ~3ms per embed)
store = MemoryStore("postgresql://...")

# OpenAI: higher quality, costs money, ~100ms per embed
store = MemoryStore("postgresql://...", embedder=OpenAIEmbedder())
store = MemoryStore("postgresql://...", embedder=OpenAIEmbedder(
    model="text-embedding-3-large",  # 3072 dims
))

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

The database schema auto-adapts to the embedding dimensionality. Just switch the embedder and go.

---

## More Examples

### Personal assistant that remembers preferences

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

### Customer support bot with history

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

### Fact versioning — when things change

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

### Background consolidation

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
- **Dedup** — merges near-identical memories (cosine > 0.92) using numpy matmul
- **Decay** — reduces importance of memories not accessed in 7/30 days
- **Expire** — soft-deletes raw chunks past their 30-day TTL
- **Promote** — distills raw conversation chunks into insights (with optional LLM)

### REST API

Mount on any FastAPI app — 17 endpoints out of the box:

```python
from unforget.api import create_memory_router

app.include_router(create_memory_router(store), prefix="/v1/memory")
# write, recall, auto-recall, ingest, list, get, update, delete,
# bulk-delete, supersede, timeline, chain, history, stats, consolidate
```

---

## Performance

Unforget ships with ONNX Runtime support for the embedding model, cutting inference time in half. The embedding cache eliminates redundant computation for repeated content.

| Operation | Latency | Notes |
|-----------|---------|-------|
| `write()` | ~7ms | ONNX embed + single SQL insert |
| `recall()` | ~25ms | Embed + 4-channel CTE + rerank |
| `auto_recall()` | ~25ms | recall + format for system prompt |
| `write_batch(20)` | ~65ms | Batch embedding + batch insert |
| Cache hit (recall) | <0.1ms | TTL cache for repeated queries |
| Cache hit (embed) | <0.1ms | LRU cache by content hash |

### Autoimprove (experimental)

Unforget includes an autoresearch-style optimization loop inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). A Python script runs in a loop on your machine, using Claude Code as the brain:

```
benchmark → analyze bottleneck → propose ONE optimization → test → benchmark → keep or revert
```

The loop has already produced 6 merged optimizations across 14 cycles:

| Optimization | Impact |
|---|---|
| Consolidation: numpy matmul (replaces N HNSW queries) | 144ms → 16ms |
| Embedding LRU cache (SHA-256 keyed, 4096 entries) | Cache hits: <0.1ms |
| ONNX Runtime for embedder | Embed: 9ms → 3ms |
| Store: CTE insert + audit trail (single round trip) | ~3ms saved per write |
| Retrieval: conditional entity CTE | ~1ms saved |
| Reranker: batch optimization | +6 benchmark points |

Run it yourself:

```bash
# Single cycle
python autoimprove/autoimprove.py --once

# Continuous loop (every 4 hours)
python autoimprove/autoimprove.py --interval 4

# Quick mode for faster iteration
python autoimprove/autoimprove.py --once --quick
```

The autoimprove agent follows strict rules: one change per cycle, max 2 files, all 265 tests must pass, benchmark score must improve by at least 1 point. Failed experiments are automatically reverted. See `autoimprove/program.md` for the full constraint set.

---

## Infrastructure

- **Database**: PostgreSQL + pgvector — one `docker compose up`
- **Embedding**: `all-MiniLM-L6-v2` — ONNX Runtime (default) or PyTorch fallback
- **Reranking**: `ms-marco-MiniLM-L-6-v2` cross-encoder — local, CPU, ~10ms for 12 pairs
- **No external APIs** required for core functionality
- **Python**: 3.11+

```bash
# Start PostgreSQL + pgvector
docker compose up -d

# Install
pip install unforget[openai]     # or unforget[anthropic] or unforget[api]
```

### ONNX Setup (optional, recommended)

ONNX Runtime makes embedding 2-3x faster. Export the models once:

```bash
pip install optimum[onnxruntime]

python -c "
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
m = ORTModelForFeatureExtraction.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', export=True)
m.save_pretrained('models/embedder-onnx')
AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').save_pretrained('models/embedder-onnx')
"
```

Unforget auto-detects `models/embedder-onnx/model.onnx` and uses it. Falls back to PyTorch if not found.

---

## Install

```bash
pip install unforget                # core only
pip install unforget[openai]        # + OpenAI SDK wrapper
pip install unforget[anthropic]     # + Anthropic SDK wrapper
pip install unforget[api]           # + FastAPI router
pip install unforget[spacy]         # + better entity extraction
```

---

## License

Apache 2.0
