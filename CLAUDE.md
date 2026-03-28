# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Unforget

Unforget is a Python library for AI agent memory backed by PostgreSQL + pgvector. It provides zero-LLM writes (~7ms), 4-channel hybrid retrieval (semantic + BM25 + entity + temporal via a single SQL CTE with Reciprocal Rank Fusion), multi-tenant scoping (org_id + agent_id), and first-class OpenAI/Anthropic SDK wrappers.

## Common Commands

```bash
# Install (editable + dev deps)
pip install -e ".[dev]"

# Run all tests (requires PostgreSQL via docker compose up -d)
pytest tests/

# Run tests without database
UNFORGET_SKIP_DB_TESTS=1 pytest tests/

# Run a single test
pytest tests/test_store.py::test_name -v

# Lint
ruff check src/

# Type check
mypy src/

# Benchmarks
python -m benchmarks.bench_performance
python -m benchmarks.bench_performance --quick  # fast subset

# Database
docker compose up -d  # pgvector on port 5433

# Release
./scripts/release.sh patch|minor|major
```

## Architecture

### Core Python package: `src/unforget/`

- **store.py** — `MemoryStore`: main orchestrator. All write/recall/consolidate operations go through here.
- **retrieval.py** — 4-channel retrieval in a single SQL query via CTEs, fused with RRF. Boosted by memory type (insight > event > raw).
- **embedder.py** — `Embedder` (ONNX/sentence-transformers), `OpenAIEmbedder`. Embedding cache is LRU by SHA-256 content hash.
- **reranker.py** — Cross-encoder reranking (ms-marco-MiniLM-L-6-v2).
- **consolidation.py** — Background dedup (cosine > 0.92), importance decay, TTL expiry, raw-to-insight promotion. Runs async via `ConsolidationScheduler`.
- **tools.py** — `MemoryToolExecutor` with 5 LLM-callable tools (store, search, list, update, forget).
- **api.py** — `create_memory_router()` returns a FastAPI router with 17 endpoints under `/v1/memory/`.
- **integrations/** — `wrap_openai` and `wrap_anthropic` monkey-patch SDK clients to inject memory tools, auto-recall, and auto-ingest.
- **scoped.py** — `ScopedMemory`: pre-binds org_id/agent_id for cleaner single-agent usage (`store.bind(...)`).
- **temporal.py** — Versioning: supersede chains, timeline queries, soft deletes via `valid_to`.
- **schema.py** — Idempotent `ensure_schema()` for tables + indexes (HNSW, GIN, B-tree).
- **types.py** — Pydantic models: `MemoryType` (RAW/EVENT/INSIGHT), `WriteItem`, `MemoryResult`, etc.

### Database

PostgreSQL 16 + pgvector. Two tables: `memory` (main) and `memory_history` (audit trail). Key indexes: HNSW on embedding, GIN on tsvector and entities array, B-tree on (org_id, agent_id) scoping.

### Other directories

- **tests/** — pytest + pytest-asyncio (`asyncio_mode = "auto"`). DB tests use the `store` fixture from conftest.py, which creates an isolated MemoryStore and truncates after each test. Set `UNFORGET_SKIP_DB_TESTS=1` to skip DB-dependent tests.
- **benchmarks/** — Performance suite targeting latency (auto_recall ~30-55ms, write ~10-15ms). Composite scoring weighted by real-user hot paths.
- **autoimprove/** — Autonomous optimization agent: benchmark -> analyze bottleneck -> edit max 2 files in src/ -> test -> re-benchmark. Must not change public API, schema, tests, or benchmarks.
- **demo/** — FastAPI + WebSocket interactive demo (`python demo/server.py`).
- **docs/** — Nextra documentation site (Next.js 15). See `docs/CLAUDE.md` for Next.js-specific guidance.
- **website/** — Next.js 15 marketing site with Tailwind.

## LoCoMo Benchmark

`benchmarks/locomo/` — Evaluates Unforget on the LoCoMo (Long Conversation Memory) benchmark (ACL 2024). Tests accuracy across single-hop, temporal, multi-hop, and open-domain questions.

```bash
# Quick test
UNFORGET_TEST_DATABASE_URL="postgresql://unforget:unforget@localhost:5433/unforget" \
  python -m benchmarks.locomo.run --conversations 1 --limit 10 --ingest-mode hybrid --recall-limit 15

# Full benchmark (all 10 conversations, ~1540 questions, ~30-45 min)
UNFORGET_TEST_DATABASE_URL="postgresql://unforget:unforget@localhost:5433/unforget" \
  python -m benchmarks.locomo.run --ingest-mode hybrid --recall-limit 15
```

Best config found so far: `--ingest-mode hybrid --recall-limit 15` with gpt-4.1-mini.

## Roadmap

See `ROADMAP.md` for the prioritized feature roadmap, based on competitive analysis against Mem0, Zep, Letta, Cognee, Memori, and others. Key priorities: date-aware retrieval, fact extraction ingest mode, TypeScript SDK, CLI tool, MCP server.

## Competitive Context

Unforget's key differentiators vs competitors: zero-LLM writes (~7ms vs 500ms+), 4-channel hybrid retrieval, PostgreSQL-only (no Neo4j/Qdrant), built-in FastAPI router + LLM tools, background consolidation, temporal versioning. Main gaps: no TypeScript SDK, no CLI, no published LoCoMo benchmark (in progress), no entity relationship graph.

## Maintenance

This file must be kept up to date. After every major code change, architectural decision, or critical fix, update the relevant sections of this CLAUDE.md so future Claude Code sessions have accurate context and can provide better assistance. If a new pattern, module, or convention is introduced, document it here.

## Code Style

- Python 3.11+, line length 100
- Ruff rules: E, F, I, N, W, UP (ignoring E501, N818, E402)
- Async-first: all store operations are async
- Test DB URL defaults to `postgresql://unforget:unforget@localhost:5432/unforget`; docker-compose exposes port 5433 externally
