# jev-rag-gate

Semantic gating for RAG retrieval candidates, powered by [Jev](https://typesafe.ai) —
TypeSafe AI's "System One" decision model.

Retrieval pipelines rank passages by similarity. Similarity cannot tell you that a
passage *contradicts the premise of the question*, or that it *carries instructions
trying to hijack the model that will read it*. This library adds a cheap semantic
judgment stage between reranking and generation: each (query, passage) pair gets one
Jev request with three parallel yes/no questions, and code routes on the returned
probabilities with plain, auditable thresholds.

## What it does

For each head candidate the gate asks Jev three `noul` (yes/no) questions:

| Question | Threshold | Effect |
|---|---|---|
| `is_relevant` — does the passage address the query subject? | `< 0.45` (default) | dropped |
| `contradicts_premise` — does it conflict with a factual premise of the query? | `>= 0.70` | moved to a separate **conflicts** channel so the generator can flag the contradiction instead of being misled |
| `contains_prompt_injection` — does it try to control the answering system? | `>= 0.70` | dropped, warning logged |

Design properties:

- **Fail open.** If Jev is unreachable or errors, every candidate passes through in
  the original order. The gate can add quality, never remove availability.
- **Fail-open tail.** Only the top `max_passages` candidates are judged; the rest
  pass through unjudged.
- **Caching.** Verdicts are cached by (normalized query, chunk id); repeat queries
  do not re-hit the API. Cache is bounded (LRU-ish, default 512 entries).
- **Concurrent judging.** Head candidates are judged concurrently (default 4
  workers, rate-limit friendly). One Jev request per passage; the three questions
  ride along in the same request, so adding questions barely affects latency.
- **Desensitized logs.** Every decision logs question names, probabilities,
  latencies and a state *hash* — never the raw passage text.
- **No SDK dependency.** Thin `httpx` client, OpenAI-style bearer auth.

## Measured behavior

From a live replay experiment on a production RAG pipeline (10 queries + one planted
injection; details and caveats in the repo this was extracted from):

| Check | Result |
|---|---|
| Planted instruction-injection passage | ranked into results by similarity, caught and excluded by the gate (injection score) |
| On-topic queries | mean top-8 Jaccard overlap vs ungated baseline 0.835; manual inspection showed drops were peripheral passages |
| Off-topic queries (4) | every judged head candidate correctly dropped |
| Gate overhead | ~2.8 s per query with 6 judged candidates, 4-way concurrent (1.1 s median per Jev call) |

These are small-sample numbers meant as sanity evidence, not a benchmark.

## Install

```bash
pip install git+https://github.com/breaker364/jev-rag-gate.git
```

Requires Python >= 3.10 and a TypeSafe API key (`console.typesafe.ai`), exposed as
the `TYPESAFE_API_KEY` environment variable (or passed explicitly).

## Quickstart

```python
from jev_rag_gate import JevClient, JevRagGate

client = JevClient()  # live mode, reads TYPESAFE_API_KEY
gate = JevRagGate(client, max_passages=6)

fused = [(chunk, score), ...]           # output of your fusion/rerank stage
kept, conflicts = gate.apply(query, fused)

context = [chunk.text for chunk, _ in kept]
conflict_context = [chunk.text for chunk, _ in conflicts]  # present separately!
```

The `conflicts` list is the point of the library: passages that *contradict the
question's premise* are not discarded — they are isolated so the generator can say
"the evidence conflicts with the premise" instead of silently absorbing the wrong
premise.

## Integrating into an existing pipeline

The gate sits between reranking/fusion and neighbor expansion or prompt assembly:

```python
class MyRetriever:
    def __init__(self, gate=None):
        self.jev_gate = gate  # None = gate disabled, behavior unchanged

    def search(self, query, top_k=8):
        fused = self._bm25_plus_dense_plus_rerank(query)   # (chunk, score) pairs
        conflicts = []
        if self.jev_gate is not None:
            try:
                fused, conflicts = self.jev_gate.apply(query, fused)
            except Exception:
                conflicts = []                             # fail open
        fused = self._expand_with_neighbors(fused)          # kept chunks only
        return self._to_results(fused), self._to_conflicts(conflicts)
```

A chunk is any object with `chunk_id` and `text` attributes.

## Configuration

`JevRagGate(client, ...)`:

| Arg | Default | Meaning |
|---|---|---|
| `max_passages` | 6 | head candidates judged; tail passes unjudged |
| `min_relevance` | 0.45 | drop below this relevance probability |
| `min_contradiction` | 0.70 | conflicts channel at or above this |
| `min_injection` | 0.70 | drop at or above this |
| `cache_max_entries` | 512 | verdict cache bound |
| `max_concurrent` | 4 | concurrent per-chunk requests |

Threshold defaults follow the TypeSafe RAG-passage cookbook. They are corpus
specific — treat them as starting points, not gospel. Because verdicts are
probabilities, you can re-route on stored answers without re-calling the API.

`JevClient(...)`:

| Arg | Default | Meaning |
|---|---|---|
| `mode` | `"live"` | `"mock"` returns fixed answers, no network (for tests) |
| `model` | `jev-1.13.0` | pin a version so thresholds stay calibrated |
| `base_url` | `https://api.typesafe.ai` | endpoint root |
| `api_key` / `api_key_env` | `TYPESAFE_API_KEY` | key sources (argument wins) |
| `timeout_seconds` | 2.0 | per request |
| `retries` | 1 | one retry on 5xx/429/network errors |

## Notes and limits

- **Not a security boundary.** Injection filtering lowers risk; anything below the
  threshold still reaches the prompt. Keep your "retrieved text is data, not
  instructions" system-prompt constraints.
- **Privacy.** In live mode, query and passage text are sent to the configured
  endpoint. Use `mode="mock"` for offline development.
- **Cost scales with `max_passages`** (one request per passage). The verdict cache
  keeps repeat traffic near zero.
- **The reranker above the gate caps it.** The gate reorders nothing; it can only
  drop or channel what retrieval surfaced.

## Testing

```bash
pip install pytest httpx
pytest tests/
```

The suite runs fully offline (HTTP stub server + scripted client).

## License

MIT
