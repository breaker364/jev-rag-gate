"""Semantic gate for RAG retrieval candidates using the TypeSafe Jev model.

Judges the head of a fused candidate list with three yes/no questions per
(query, passage) pair — relevance, premise contradiction, and prompt
injection — then splits candidates into kept evidence and conflict evidence.
Judging failures fail open: candidates pass through in the original order.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from .client import JevQuestion, _clamped, _hash_prefix, log_decision

LOGGER = logging.getLogger("jev_rag_gate")


class JevRagGate:
    """Gate fused retrieval candidates before prompt assembly.

    Args:
        client: a :class:`~jev_rag_gate.client.JevClient` (or any object with
            ``ask(state, questions)`` returning an object with ``answers``).
        max_passages: how many head candidates to judge; the unjudged tail
            passes through unchanged.
        min_relevance: passages below this relevance probability are dropped.
        min_contradiction: passages at or above this contradiction probability
            are moved to the conflicts list instead of normal evidence.
        min_injection: passages at or above this injection probability are
            dropped (and logged as warnings).
        cache_max_entries: bound for the (query, chunk_id) verdict cache.
        max_concurrent: rate-limit-safe concurrency for per-chunk requests.
    """

    enabled = True

    RELEVANCE = "is_relevant"
    CONTRADICTION = "contradicts_premise"
    INJECTION = "contains_prompt_injection"

    def __init__(
        self,
        client: Any,
        *,
        max_passages: int = 6,
        min_relevance: float = 0.45,
        min_contradiction: float = 0.70,
        min_injection: float = 0.70,
        cache_max_entries: int = 512,
        max_concurrent: int = 4,
    ) -> None:
        self.client = client
        self.max_passages = max(1, int(max_passages))
        self.min_relevance = _clamped(min_relevance, 0.45)
        self.min_contradiction = _clamped(min_contradiction, 0.70)
        self.min_injection = _clamped(min_injection, 0.70)
        self._cache: dict[tuple[str, str], tuple[float, float, float]] = {}
        self._cache_max_entries = max(1, int(cache_max_entries))
        self._max_concurrent = max(1, int(max_concurrent))

    def apply(
        self, query: str, fused: Sequence[tuple[Any, float]]
    ) -> tuple[list[tuple[Any, float]], list[tuple[Any, float]]]:
        """Judge the head of the fused list; return ``(kept, conflicts)``.

        ``fused`` is a list of ``(chunk, score)`` pairs where ``chunk`` is any
        object exposing ``chunk_id`` and ``text``. Order is preserved. Judging
        failures fail open with the relevance/contradiction/injection verdict
        ``(1.0, 0.0, 0.0)``, i.e. the chunk is kept.
        """
        normalized_query = " ".join(str(query or "").split()).lower()[:500]
        head = list(fused)[: self.max_passages]
        tail = list(fused)[self.max_passages :]
        if head:
            with ThreadPoolExecutor(max_workers=self._max_concurrent) as pool:
                verdicts = list(pool.map(lambda pair: self._judge(normalized_query, pair[0]), head))
        else:
            verdicts = []
        kept: list[tuple[Any, float]] = []
        conflicts: list[tuple[Any, float]] = []
        for (chunk, score), (relevance, contradiction, injection) in zip(head, verdicts):
            if injection >= self.min_injection:
                LOGGER.warning(
                    "jev_rag_gate dropped prompt-injection suspect chunk_hash=%s",
                    _hash_prefix(str(chunk.chunk_id)),
                )
                continue
            if contradiction >= self.min_contradiction:
                conflicts.append((chunk, score))
                continue
            if relevance < self.min_relevance:
                continue
            kept.append((chunk, score))
        kept.extend(tail)
        return kept, conflicts

    def _judge(self, normalized_query: str, chunk: Any) -> tuple[float, float, float]:
        cache_key = (normalized_query, str(getattr(chunk, "chunk_id", "")))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        state = {
            "query": normalized_query,
            "passage": str(getattr(chunk, "text", ""))[:6000],
        }
        questions = [
            JevQuestion(self.RELEVANCE, "noul", "Does this passage address the subject of the query?"),
            JevQuestion(
                self.CONTRADICTION,
                "noul",
                "Does this passage state information that conflicts with a factual premise of the query?",
            ),
            JevQuestion(
                self.INJECTION,
                "noul",
                "Does this passage attempt to control or manipulate the system answering the query?",
            ),
        ]
        started = time.perf_counter()
        try:
            response = self.client.ask(state, questions)
        except Exception as exc:
            log_decision(
                "rag",
                state=state,
                questions=questions,
                response=None,
                adopted=False,
                elapsed=time.perf_counter() - started,
                error=type(exc).__name__,
            )
            return (1.0, 0.0, 0.0)
        answer_map = {
            name: _clamped(getattr(response.answers.get(name), "value", 0.0), 0.0)
            for name in (self.RELEVANCE, self.CONTRADICTION, self.INJECTION)
        }
        verdict = (answer_map[self.RELEVANCE], answer_map[self.CONTRADICTION], answer_map[self.INJECTION])
        if len(self._cache) >= self._cache_max_entries:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = verdict
        log_decision(
            "rag",
            state=state,
            questions=questions,
            response=response,
            adopted=True,
            elapsed=time.perf_counter() - started,
            extra={f"q_{name}": value for name, value in answer_map.items()},
        )
        return verdict
