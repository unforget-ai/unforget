# Roadmap

Prioritized feature roadmap for Unforget, based on competitive analysis (March 2026)
against Mem0, Zep/Graphiti, Letta, Cognee, Memori, MemOS, and LangMem.

---

## P0 — Do Now (Week of 2026-03-28)

### Graduate from Alpha to Beta (v0.3.0)
Codebase quality, test coverage (19 test files, 3500+ LOC), and documentation
are beyond alpha grade. The "Alpha" classifier is suppressing adoption —
enterprise teams and serious builders skip alpha libraries. Bump to Beta, tag
v0.3.0, and add a stability commitment to the README (stable public API,
backward-compatible schema migrations).

### Publish LoCoMo benchmark results
Full benchmark built in `benchmarks/locomo/`. Publish results with comparison
table against Mem0 (66.9%), Zep (75.1%), Memori (82.0%). Write Medium article
with the narrative: "Fastest writes AND competitive recall accuracy."

### Competitive comparison page
Unforget's ~7ms writes and ~30-55ms recall are exceptional vs alternatives
(Mem0, Zep, LangMem) that require LLM calls on write (500ms+). Create a
comparison page on the docs/website covering latency, cost (zero per-write API
cost), and architecture simplicity (single PostgreSQL dependency).

---

## P0.5 — LoCoMo Score Push (80.8% → 88-91%)

Current baseline: 80.8% overall (single_hop 71.6%, multi_hop 62.5%,
temporal 82.5%, open_domain 85.4%). Tested with `--ingest-mode hybrid
--recall-limit 15` on 2026-03-28.

### 1. Association linking + pull (multi_hop +10-15%)
**Why**: Multi-hop is the worst category at 62.5%. These questions require
combining 2-3 memories (e.g., "What restaurant did the person from Brooklyn
recommend?"). Retrieval finds one relevant memory but misses the connecting
fact. Association pull means retrieving memory A automatically boosts memory B
if they co-occurred in the same conversation window. This is how human memory
works — one memory triggers a cluster. Adds a `memory_associations` table,
a scheduler job to build links, and a pull mechanism in the recall path.

### 2. Multi-query decomposition at recall (multi_hop +8-12%)
**Why**: Multi-hop questions contain multiple sub-questions fused into one.
The embedding of the composite query matches neither sub-fact well. At recall
time, use a fast LLM call (~200ms) to split the query into sub-queries,
retrieve for each, merge results. Directly targets the 62.5% multi_hop score
without changing the write path or schema.

### 3. Fact-extraction ingest mode (single_hop +5-8%)
**Why**: Single_hop is 71.6% — simple factual lookups are failing because
facts are buried inside long conversation chunks. A `"extract"` ingest mode
pulls discrete claims ("Person X lives in Y") as individual insight memories,
making them directly retrievable. Already on the P1 roadmap — promoting
priority due to benchmark impact.

### 4. Conflict resolution (single_hop +3-5%)
**Why**: Some single_hop failures happen when a user updated a fact (moved
cities, changed jobs) and both old and new memories exist. Retrieval surfaces
the stale one. A conflict resolver detects same-topic contradictions, scores
by recency and usage, and supersedes the outdated memory. Runs async in the
consolidation scheduler. Part of the MJL (Memory Judgment Layer) architecture.

### 5. Entity channel weight tuning (single_hop +2-3%)
**Why**: When a query mentions "Sarah" and a memory mentions "Sarah", entity
overlap should be a strong signal. The current RRF weight for the entity
channel may be too low relative to semantic similarity. Tuning this is a
config change with measurable impact — run LoCoMo with different entity
weights and pick the best.

---

## P1 — High Impact (Next 2 Weeks)

### User-level memory isolation (security + multi-tenant)
Add `user_id` as a first-class column on the memory table. Current scoping is
`(org_id, agent_id)` which means all users of an agent share the same memory
pool — a data leak in production. With `user_id`, scoping becomes a 3-level
hierarchy:
- `org + agent + user` → personal memories (isolated per user)
- `org + agent` (shared=true) → agent knowledge base (visible to all users)
- `org + user` → user profile across agents (for handoffs)

Update unique constraint to `UNIQUE(org_id, agent_id, user_id, content)`.
Enforce `user_id` required on all write/recall API calls. Scope association
pull within `(org, agent, user)` to prevent cross-user linking. Backward
compatible: existing code with `user_id=None` keeps working.

### Distribution: get 5-10 real agent builders using Unforget
Engineering is strong; the bottleneck is distribution. Actions:
- Post "Show HN" on Hacker News and share on r/LocalLLaMA, AI agent Discords.
- Reach out to open-source agent framework maintainers (CrewAI, AutoGen,
  LangGraph) about integration or endorsement.
- Add a one-command demo experience so people can try it in 30 seconds:
  `pip install unforget[demo] && python -m unforget.demo`

### Date-aware retrieval (inspired by Hindsight's temporal filtering)
The temporal channel currently ranks by `accessed_at` (recency), not by dates
mentioned in memory content. Questions like "When did X happen in June?" suffer.
Extract dates from content at write time, store in a `content_date` column, and
add a date-range filter to the temporal CTE. Hindsight does this and it's a key
reason they score well on temporal questions in LongMemEval.

### Entity profiles
Accumulate facts per entity in a lightweight table (e.g., "Caroline: transgender,
lives in Austin, likes art"). Query during recall for entity-centric questions.
Similar to Hindsight's entity profiles but without requiring LLM on write —
build profiles incrementally from extracted entities during consolidation.

### Fact extraction ingest mode
Add an `"extract"` ingest mode that uses an LLM to pull discrete facts from
conversations (e.g., "Caroline is transgender", "Melanie likes swimming") and
stores each as an individual insight memory. This is Mem0's core feature and
would significantly improve single-hop recall accuracy.

### Reflect operation
On-demand LLM reasoning over stored memories (like Hindsight's `reflect()`).
Given a query, retrieve relevant memories and use an LLM to synthesize a
deeper answer. Useful for multi-hop questions and building mental models.

### Recall explain — "why this memory?"
Add `explain=True` parameter to `recall()`. Returns per-channel ranks,
RRF contributions, type boost, and association pull details for each result.
Developers can't debug retrieval without knowing which channels contributed.
Also supports `explain_misses=5` to show why expected memories didn't rank.

### Memory inspector — full lifecycle audit
`memory.inspect(id)` returns the complete picture: current state, history
timeline (created → accessed → promoted → decayed), association graph
neighbors with strength, supersession chain (version history), and decay
projection (when will importance drop below threshold). One call answers
"what happened to this memory?"

### Dry-run consolidation
`memory.consolidate(dry_run=True)` runs the full consolidation logic in a
read-only transaction (rolled back). Shows what would be merged, promoted,
decayed, and expired — without changing anything. Critical for developers
tuning consolidation thresholds.

### pgbox — embedded PostgreSQL (replaces pgserver)
Own package (`pip install pgbox`) bundling PostgreSQL 17.4 + pgvector 0.8.0.
Replaces unmaintained pgserver (stuck on PG 16.2, Python 3.9-3.12). Typed API
with `get_server()`, `enable_extension()`, `create_database()`, health checks.
Enables `unforget init` and `pip install unforget[embed]` zero-config experience.

### CLI tool
```
unforget init                                        # start embedded PG
unforget demo                                        # launch demo2 chat
unforget add "User prefers dark mode" --org acme --agent bot
unforget search "preferences" --org acme --agent bot
unforget list --org acme --agent bot
unforget stats --org acme --agent bot
```
Low effort, high DX impact for demos and quick testing.

---

## P2 — Important (Next Month)

### TypeScript/JavaScript SDK
Thin REST client over the existing FastAPI router. Mem0, Zep, Letta, and Memori
all have JS/TS SDKs. Critical for reaching the agent ecosystem.

### MCP server
Expose the 5 memory tools via Model Context Protocol for Claude/Cursor
integration. Graphiti already has one. Growing ecosystem demand.

### Broader entity extraction
`KNOWN_ENTITIES` is ~100 tech terms. For general conversation memory, add
person names, places, organizations, and dates via lightweight NER (regex
patterns for names/dates, optional spaCy for production).

### Persist metadata field
`WriteItem.metadata` is defined but never saved to the database. Persist it as
a JSONB column to enable structured filtering (speaker names, session IDs,
custom attributes).

---

## P3 — Strategic (Next Quarter)

### Entity relationship modeling
Entities are extracted but flat. Add a lightweight `entity_relationships` table
in PostgreSQL with (subject, predicate, object) triples. No Neo4j needed —
recursive CTEs can traverse relationships. Closes the gap with Zep/Graphiti.

### Web dashboard
Read-only memory browser for debugging and demos. Mem0 and Memori both have
web UIs. Could be a simple Next.js app using the FastAPI router.

### Additional benchmarks
Consider LongMemEval, PrefEval, PersonaMem — MemOS publishes on all of these.
More benchmarks = stronger credibility.

---

## Housekeeping

### Deduplicate `_row_to_item`
Identical implementations in `store.py` and `temporal.py`. Extract to shared
utility.

### Async OpenAIEmbedder
`OpenAIEmbedder` uses synchronous `OpenAI` client which blocks the event loop
in async contexts. Switch to `AsyncOpenAI`.
