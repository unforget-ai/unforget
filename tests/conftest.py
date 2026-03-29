"""Test fixtures for unforget."""

import os

import pytest
import pytest_asyncio

from unforget import MemoryStore

DATABASE_URL = os.environ.get(
    "UNFORGET_TEST_DATABASE_URL",
    "postgresql://unforget:unforget@localhost:5432/unforget",
)

# Skip all integration tests if no database is available
requires_db = pytest.mark.skipif(
    os.environ.get("UNFORGET_SKIP_DB_TESTS", "0") == "1",
    reason="UNFORGET_SKIP_DB_TESTS=1",
)


@pytest_asyncio.fixture
async def store():
    """Create a MemoryStore connected to the test database.

    Cleans up all test data after each test.
    """
    s = MemoryStore(database_url=DATABASE_URL)
    await s.initialize()
    yield s
    # Cleanup: delete all test data
    await s.pool.execute("DELETE FROM memory_associations")
    await s.pool.execute("DELETE FROM memory_history")
    await s.pool.execute("DELETE FROM memory")
    await s.close()
