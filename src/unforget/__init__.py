"""unforget — Fast, zero-LLM memory for AI agents."""

from unforget.consolidation import ConsolidationReport as ConsolidationReport
from unforget.embedder import BaseEmbedder as BaseEmbedder
from unforget.embedder import Embedder as Embedder
from unforget.embedder import OpenAIEmbedder as OpenAIEmbedder
from unforget.quotas import RateLimitExceeded as RateLimitExceeded
from unforget.retrieval import RetrievalConfig as RetrievalConfig
from unforget.scheduler import ConsolidationScheduler as ConsolidationScheduler
from unforget.scoped import ScopedMemory as ScopedMemory
from unforget.store import MemoryQuotaExceeded as MemoryQuotaExceeded
from unforget.store import MemoryStore as MemoryStore
from unforget.tools import MEMORY_TOOLS as MEMORY_TOOLS
from unforget.tools import MemoryToolExecutor as MemoryToolExecutor
from unforget.types import HistoryAction as HistoryAction
from unforget.types import MemoryHistoryEntry as MemoryHistoryEntry
from unforget.types import MemoryItem as MemoryItem
from unforget.types import MemoryResult as MemoryResult
from unforget.types import MemoryStats as MemoryStats
from unforget.types import MemoryType as MemoryType
from unforget.types import WriteItem as WriteItem

from importlib.metadata import version as _version

__version__ = _version("unforget")

__all__ = [
    # Core
    "MemoryStore",
    "ScopedMemory",
    # Embedders
    "BaseEmbedder",
    "Embedder",
    "OpenAIEmbedder",
    # Types
    "MemoryType",
    "MemoryItem",
    "MemoryResult",
    "MemoryStats",
    "MemoryHistoryEntry",
    "WriteItem",
    "HistoryAction",
    # Errors
    "MemoryQuotaExceeded",
    "RateLimitExceeded",
    # Config
    "RetrievalConfig",
    "ConsolidationReport",
    "ConsolidationScheduler",
    # Tools
    "MemoryToolExecutor",
    "MEMORY_TOOLS",
]
