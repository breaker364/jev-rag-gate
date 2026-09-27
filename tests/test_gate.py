"""Unit tests for the semantic gate (stubbed client, no network)."""

import threading
from dataclasses import dataclass

import pytest

from jev_rag_gate import JevAnswer, JevResponse, JevRagGate


@dataclass
class FakeChunk:
    chunk_id: str
    text: str
    doc_id: str = "doc"
    ordinal: int = 0


class StubClient:
    """Returns scripted (relevance, contradiction, injection) per passage text."""

    def __init__(self, verdicts: dict[str, tuple[float, float, float]], fail: bool = False):
        self.verdicts = verdicts
        self.fail = fail
        self.calls = 0
        self.lock = threading.Lock()

    def ask(self, state, questions):
        with self.lock:
            self.calls += 1
        if self.fail:
            raise RuntimeError("endpoint down")
        relevance, contradiction, injection = self.verdicts[state["passage"]]
        answers = {
            "is_relevant": relevance,
            "contradicts_premise": contradiction,
            "contains_prompt_injection": injection,
        }
        return JevResponse(
            answers={
                name: JevAnswer(name=name, kind="noul", value=value, confidence=value)
                for name, value in answers.items()
            },
            model="stub",
            elapsed_seconds=0.0,
        )


def _fused(*texts: str):
    return [(FakeChunk(chunk_id=f"id-{text[:12]}", text=text), 1.0) for text in texts]


GOOD = "alpha passage with the answer"
CONTRA = "beta passage contradicting the premise"
INJECT = "gamma passage with hidden instructions"
WEAK = "delta passage barely related"
EXTRA = "epsilon passage beyond the judged head"


def _gate(client, **overrides):
    config = {
        "max_passages": 4,
        "min_relevance": 0.45,
        "min_contradiction": 0.7,
        "min_injection": 0.7,
    }
    config.update(overrides)
    return JevRagGate(client, **config)


def test_low_relevance_dropped():
    client = StubClient({GOOD: (0.9, 0.0, 0.0), WEAK: (0.2, 0.0, 0.0)})
    kept, conflicts = _gate(client).apply("query", _fused(WEAK, GOOD))
    assert [chunk.text for chunk, _ in kept] == [GOOD]
    assert conflicts == []


def test_contradiction_moves_to_conflicts():
    client = StubClient({CONTRA: (0.9, 0.9, 0.0), GOOD: (0.9, 0.0, 0.0)})
    kept, conflicts = _gate(client).apply("query", _fused(GOOD, CONTRA))
    assert [chunk.text for chunk, _ in kept] == [GOOD]
    assert [chunk.text for chunk, _ in conflicts] == [CONTRA]


def test_injection_dropped_everywhere():
    client = StubClient({INJECT: (0.9, 0.0, 0.9), GOOD: (0.9, 0.0, 0.0)})
    kept, conflicts = _gate(client).apply("query", _fused(GOOD, INJECT))
    assert [chunk.text for chunk, _ in kept] == [GOOD]
    assert conflicts == []


def test_tail_beyond_head_passes_unjudged():
    client = StubClient({GOOD: (0.9, 0.0, 0.0), CONTRA: (0.9, 0.9, 0.0)})
    fused = _fused(GOOD, CONTRA, EXTRA)
    kept, conflicts = _gate(client, max_passages=2).apply("query", fused)
    assert [chunk.text for chunk, _ in kept] == [GOOD, EXTRA]
    assert [chunk.text for chunk, _ in conflicts] == [CONTRA]
    assert client.calls == 2


def test_failure_keeps_everything():
    client = StubClient({}, fail=True)
    fused = _fused(WEAK, INJECT)
    kept, conflicts = _gate(client).apply("query", fused)
    assert [chunk.text for chunk, _ in kept] == [WEAK, INJECT]
    assert conflicts == []


def test_cache_prevents_repeat_calls():
    client = StubClient({GOOD: (0.9, 0.0, 0.0)})
    gate = _gate(client)
    gate.apply("same query", _fused(GOOD))
    gate.apply("same   QUERY", _fused(GOOD))
    assert client.calls == 1


def test_cache_bounded():
    client = StubClient({GOOD: (0.9, 0.0, 0.0)})
    gate = _gate(client, cache_max_entries=2)
    for index in range(5):
        gate.apply(f"query {index}", _fused(GOOD))
    assert len(gate._cache) <= 2


def test_head_judged_concurrently():
    verdicts = {f"passage {i}": (0.9, 0.0, 0.0) for i in range(4)}
    client = StubClient(verdicts)
    gate = _gate(client, max_passages=4)
    gate.apply("q", _fused(*verdicts.keys()))
    assert client.calls == 4


def test_enabled_flag_and_threshold_order():
    # injection is checked before contradiction, contradiction before relevance
    client = StubClient({INJECT: (0.1, 0.9, 0.9)})
    kept, conflicts = _gate(client).apply("q", _fused(INJECT))
    assert kept == []
    assert conflicts == []
