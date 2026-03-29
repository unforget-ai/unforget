# Changelog

## v0.3.0 (2026-03-29)

### Retrieval Accuracy

- **Fix: `write_batch` entity extraction** — was passing empty entities on every batch write, making the entity retrieval channel completely non-functional
- **Person name extraction** — detects names like Caroline, John Smith from conversation text
- **Date extraction** — extracts month/year mentions (May 2023, June) for temporal matching
- **Wider reranker candidate pool** — fetches limit×3 candidates instead of limit+2, so the right memory isn't missed
- **Result deduplication** — removes overlapping content from hybrid ingestion results
- **BM25 weight boost** — 0.8→1.0 for better keyword matching
- **Entity weight boost** — 0.6→0.7
- **Per-channel limit** — 8→15 for more candidates per retrieval channel

### LoCoMo Benchmark

- **80.8% overall accuracy** on the LoCoMo benchmark (ACL 2024)
- Beats Zep (75.1%) and Mem0 (66.9%) — with zero LLM calls on write
- Benchmark suite included in `benchmarks/locomo/`
- Category scores: single-hop 71.6%, temporal 82.5%, multi-hop 62.5%, open-domain 85.4%

### New Packages

- **unforget-embed** — zero-config embedded server with PostgreSQL via pgserver. `pip install unforget-embed && unforget-embed start`
- **@unforget-ai/openclaw** — OpenClaw plugin with auto-recall, auto-retain, and forget/remember intent detection. `openclaw plugins install @unforget-ai/openclaw`

### Docs & DX

- Improved README with logo, badges, and comparison table
- CLAUDE.md for Claude Code context
- OpenClaw integration docs
- PyPI publish workflow via trusted publishing

## v0.2.1 (2026-03-28)

Initial public release.

- Zero-LLM writes (~7ms)
- 4-channel hybrid retrieval (semantic + BM25 + entity + temporal) with RRF fusion
- Cross-encoder reranking
- PostgreSQL + pgvector — single database, no external services
- OpenAI and Anthropic SDK wrappers (`wrap_openai`, `wrap_anthropic`)
- FastAPI router with 17 endpoints
- 5 LLM-callable memory tools
- Background consolidation (dedup, decay, expire, promote)
- Temporal versioning (supersede, timeline, audit trail)
- Multi-tenant scoping (org_id + agent_id)
- Rate limiting and memory quotas
