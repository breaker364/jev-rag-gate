"""jev-rag-gate: semantic gating for RAG retrieval candidates, powered by Jev."""

from .client import (
    JevAnswer,
    JevClient,
    JevQuestion,
    JevResponse,
    JevUnavailable,
    log_decision,
)
from .gate import JevRagGate

__all__ = [
    "JevAnswer",
    "JevClient",
    "JevQuestion",
    "JevResponse",
    "JevRagGate",
    "JevUnavailable",
    "log_decision",
]
__version__ = "0.1.0"
