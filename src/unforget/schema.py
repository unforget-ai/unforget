"""Database schema creation and management for unforget."""

from __future__ import annotations

import asyncpg

SCHEMA_VERSION = 1

CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"

CREATE_MEMORY_TABLE = """
CREATE TABLE IF NOT EXISTS memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,

    -- Content
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'insight',
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],

    -- Retrieval indexes
    embedding VECTOR({dims}),
    search_vector TSVECTOR
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    entities TEXT[] DEFAULT ARRAY[]::TEXT[],

    -- Temporal validity
    valid_from TIMESTAMPTZ DEFAULT now(),
    valid_to TIMESTAMPTZ,
    superseded_by UUID REFERENCES memory(id),

    -- Scoring & lifecycle
    importance FLOAT DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    accessed_at TIMESTAMPTZ DEFAULT now(),
    created_at TIMESTAMPTZ DEFAULT now(),
    consolidated_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,

    -- Provenance
    source_thread_id TEXT,
    source_message TEXT,

    -- Sharing & immutability
    shared BOOLEAN DEFAULT false,
    immutable BOOLEAN DEFAULT false,

    -- Dedup
    UNIQUE(org_id, agent_id, content)
);
"""

CREATE_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS memory_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL REFERENCES memory(id) ON DELETE CASCADE,
    action TEXT NOT NULL,
    old_content TEXT,
    new_content TEXT,
    changed_at TIMESTAMPTZ DEFAULT now(),
    changed_by TEXT DEFAULT 'system'
);
"""

CREATE_ASSOCIATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS memory_associations (
    memory_a    UUID NOT NULL REFERENCES memory(id) ON DELETE CASCADE,
    memory_b    UUID NOT NULL REFERENCES memory(id) ON DELETE CASCADE,
    strength    FLOAT NOT NULL DEFAULT 1.0,
    link_type   TEXT NOT NULL DEFAULT 'co_occurrence',
    created_at  TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (memory_a, memory_b),
    CHECK (memory_a < memory_b)
);
"""

CREATE_INDEXES = [
    """CREATE INDEX IF NOT EXISTS memory_embedding_idx ON memory
       USING hnsw (embedding vector_cosine_ops)
       WITH (m = 16, ef_construction = 64);""",
    """CREATE INDEX IF NOT EXISTS memory_search_idx ON memory
       USING gin (search_vector);""",
    """CREATE INDEX IF NOT EXISTS memory_entities_idx ON memory
       USING gin (entities);""",
    """CREATE INDEX IF NOT EXISTS memory_scope_idx ON memory
       (org_id, agent_id, memory_type);""",
    """CREATE INDEX IF NOT EXISTS memory_temporal_idx ON memory
       (org_id, agent_id, valid_to NULLS FIRST);""",
    """CREATE INDEX IF NOT EXISTS memory_shared_idx ON memory
       (org_id, shared) WHERE shared = true;""",
    """CREATE INDEX IF NOT EXISTS memory_history_idx ON memory_history
       (memory_id, changed_at);""",
    # Association indexes for fast lookup from either side
    """CREATE INDEX IF NOT EXISTS memory_assoc_a_idx ON memory_associations (memory_a);""",
    """CREATE INDEX IF NOT EXISTS memory_assoc_b_idx ON memory_associations (memory_b);""",
]


async def ensure_schema(pool: asyncpg.Pool, *, dims: int = 384) -> None:
    """Create tables and indexes if they don't exist.

    Idempotent — safe to call on every startup.
    """
    async with pool.acquire() as conn:
        await conn.execute(CREATE_EXTENSION)
        await conn.execute(CREATE_MEMORY_TABLE.format(dims=dims))
        await conn.execute(CREATE_HISTORY_TABLE)
        await conn.execute(CREATE_ASSOCIATIONS_TABLE)
        for idx in CREATE_INDEXES:
            await conn.execute(idx)
