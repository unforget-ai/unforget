"""Integration tests for the FastAPI router — requires PostgreSQL + pgvector."""

import uuid

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from unforget.api import create_memory_router

from .conftest import requires_db

ORG = "test-org"
AGENT = "test-agent"


@pytest.fixture
async def app(store):
    """Create a FastAPI app with the memory router mounted."""
    app = FastAPI()
    router = create_memory_router(store)
    app.include_router(router, prefix="/v1/memory")
    return app


@pytest.fixture
async def client(app):
    """Async HTTP client for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@requires_db
class TestWriteAPI:
    async def test_write(self, client):
        resp = await client.post("/v1/memory/write", json={
            "content": "User prefers Python",
            "org_id": ORG,
            "agent_id": AGENT,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "User prefers Python"
        assert data["memory_type"] == "insight"

    async def test_write_duplicate_409(self, client):
        body = {"content": "duplicate fact", "org_id": ORG, "agent_id": AGENT}
        resp1 = await client.post("/v1/memory/write", json=body)
        assert resp1.status_code == 200
        resp2 = await client.post("/v1/memory/write", json=body)
        assert resp2.status_code == 409

    async def test_write_batch(self, client):
        resp = await client.post("/v1/memory/write/batch", json={
            "org_id": ORG,
            "agent_id": AGENT,
            "items": [
                {"content": "batch one", "org_id": ORG, "agent_id": AGENT},
                {"content": "batch two", "org_id": ORG, "agent_id": AGENT},
            ],
        })
        assert resp.status_code == 200
        assert len(resp.json()) == 2


@requires_db
class TestRecallAPI:
    async def test_recall(self, client):
        await client.post("/v1/memory/write", json={
            "content": "Deploy to Fly.io",
            "org_id": ORG,
            "agent_id": AGENT,
        })
        resp = await client.post("/v1/memory/recall", json={
            "query": "deployment target",
            "org_id": ORG,
            "agent_id": AGENT,
            "rerank": False,
        })
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) > 0
        assert "Fly" in results[0]["content"]

    async def test_auto_recall(self, client):
        await client.post("/v1/memory/write", json={
            "content": "Team uses Go",
            "org_id": ORG,
            "agent_id": AGENT,
        })
        resp = await client.post("/v1/memory/auto-recall", json={
            "query": "programming language",
            "org_id": ORG,
            "agent_id": AGENT,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "[Memory Context]" in data["context"]
        assert data["memory_count"] >= 1


@requires_db
class TestIngestAPI:
    async def test_ingest_background(self, client):
        resp = await client.post("/v1/memory/ingest", json={
            "messages": [
                {"role": "user", "content": "I work at Acme Corp. We use Python."},
                {"role": "assistant", "content": "Got it!"},
            ],
            "org_id": ORG,
            "agent_id": AGENT,
            "mode": "background",
        })
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    async def test_ingest_immediate_without_llm(self, client):
        resp = await client.post("/v1/memory/ingest", json={
            "messages": [{"role": "user", "content": "Test."}],
            "org_id": ORG,
            "agent_id": AGENT,
            "mode": "immediate",
        })
        assert resp.status_code == 400


@requires_db
class TestGetListAPI:
    async def test_get_memory(self, client):
        write_resp = await client.post("/v1/memory/write", json={
            "content": "findable fact",
            "org_id": ORG,
            "agent_id": AGENT,
        })
        memory_id = write_resp.json()["id"]

        resp = await client.get(f"/v1/memory/{memory_id}")
        assert resp.status_code == 200
        assert resp.json()["content"] == "findable fact"

    async def test_get_nonexistent_404(self, client):
        resp = await client.get(f"/v1/memory/{uuid.uuid4()}")
        assert resp.status_code == 404

    async def test_list_memories(self, client):
        await client.post("/v1/memory/write", json={
            "content": "list item one",
            "org_id": ORG,
            "agent_id": AGENT,
        })
        await client.post("/v1/memory/write", json={
            "content": "list item two",
            "org_id": ORG,
            "agent_id": AGENT,
        })

        resp = await client.get("/v1/memory/", params={
            "org_id": ORG,
            "agent_id": AGENT,
        })
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_list_with_search(self, client):
        await client.post("/v1/memory/write", json={
            "content": "Python is great for backend development",
            "org_id": ORG,
            "agent_id": AGENT,
        })
        await client.post("/v1/memory/write", json={
            "content": "JavaScript runs in browsers",
            "org_id": ORG,
            "agent_id": AGENT,
        })

        resp = await client.get("/v1/memory/", params={
            "org_id": ORG,
            "agent_id": AGENT,
            "search": "Python",
        })
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) >= 1
        assert "Python" in items[0]["content"]

    async def test_list_pagination(self, client):
        for i in range(5):
            await client.post("/v1/memory/write", json={
                "content": f"paginated item {i}",
                "org_id": ORG,
                "agent_id": AGENT,
            })

        resp = await client.get("/v1/memory/", params={
            "org_id": ORG,
            "agent_id": AGENT,
            "page": 1,
            "page_size": 2,
        })
        assert len(resp.json()) == 2

    async def test_stats(self, client):
        await client.post("/v1/memory/write", json={
            "content": "stat fact",
            "org_id": ORG,
            "agent_id": AGENT,
        })
        resp = await client.get("/v1/memory/stats/", params={
            "org_id": ORG,
            "agent_id": AGENT,
        })
        assert resp.status_code == 200
        assert resp.json()["total"] == 1


@requires_db
class TestUpdateDeleteAPI:
    async def test_update(self, client):
        write_resp = await client.post("/v1/memory/write", json={
            "content": "old content",
            "org_id": ORG,
            "agent_id": AGENT,
        })
        memory_id = write_resp.json()["id"]

        resp = await client.put(f"/v1/memory/{memory_id}", json={
            "content": "new content",
        })
        assert resp.status_code == 200
        assert resp.json()["content"] == "new content"

    async def test_delete(self, client):
        write_resp = await client.post("/v1/memory/write", json={
            "content": "delete me",
            "org_id": ORG,
            "agent_id": AGENT,
        })
        memory_id = write_resp.json()["id"]

        resp = await client.delete(f"/v1/memory/{memory_id}")
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

        resp = await client.get(f"/v1/memory/{memory_id}")
        assert resp.status_code == 404

    async def test_bulk_delete(self, client):
        await client.post("/v1/memory/write", json={
            "content": "raw chunk to delete",
            "org_id": ORG,
            "agent_id": AGENT,
            "memory_type": "raw",
        })
        await client.post("/v1/memory/write", json={
            "content": "insight to keep",
            "org_id": ORG,
            "agent_id": AGENT,
        })

        resp = await client.post("/v1/memory/bulk-delete", json={
            "org_id": ORG,
            "agent_id": AGENT,
            "memory_type": "raw",
        })
        assert resp.status_code == 200
        assert resp.json()["count"] == 1


@requires_db
class TestTemporalAPI:
    async def test_supersede(self, client):
        write_resp = await client.post("/v1/memory/write", json={
            "content": "deploys to AWS",
            "org_id": ORG,
            "agent_id": AGENT,
        })
        memory_id = write_resp.json()["id"]

        resp = await client.post(f"/v1/memory/{memory_id}/supersede", json={
            "new_content": "deploys to Fly.io",
            "org_id": ORG,
            "agent_id": AGENT,
        })
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 2
        assert items[0]["valid_to"] is not None  # old is expired
        assert items[1]["content"] == "deploys to Fly.io"

    async def test_history(self, client):
        write_resp = await client.post("/v1/memory/write", json={
            "content": "tracked fact",
            "org_id": ORG,
            "agent_id": AGENT,
        })
        memory_id = write_resp.json()["id"]

        resp = await client.get(f"/v1/memory/{memory_id}/history")
        assert resp.status_code == 200
        history = resp.json()
        assert len(history) == 1
        assert history[0]["action"] == "created"

    async def test_chain(self, client):
        w1 = await client.post("/v1/memory/write", json={
            "content": "v1 fact",
            "org_id": ORG,
            "agent_id": AGENT,
        })
        id1 = w1.json()["id"]

        s = await client.post(f"/v1/memory/{id1}/supersede", json={
            "new_content": "v2 fact",
            "org_id": ORG,
            "agent_id": AGENT,
        })
        s.json()[1]["id"]

        resp = await client.get(f"/v1/memory/{id1}/chain")
        assert resp.status_code == 200
        chain = resp.json()
        assert len(chain) == 2
        assert chain[0]["content"] == "v1 fact"
        assert chain[1]["content"] == "v2 fact"


@requires_db
class TestConsolidateAPI:
    async def test_consolidate(self, client):
        await client.post("/v1/memory/write", json={
            "content": "some fact to consolidate",
            "org_id": ORG,
            "agent_id": AGENT,
        })
        resp = await client.post("/v1/memory/consolidate", json={
            "org_id": ORG,
            "agent_id": AGENT,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "duplicates_merged" in data
        assert "memories_decayed" in data
        assert "memories_expired" in data
        assert "memories_promoted" in data
