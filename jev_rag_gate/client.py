"""Thin synchronous client for the TypeSafe Jev (System One) decision API."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import httpx

LOGGER = logging.getLogger("jev_rag_gate")

ENDPOINT_PATH = "/v1/systemone"
DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_API_KEY_ENV = "TYPESAFE_API_KEY"
DEFAULT_MODEL = "jev-1.13.0"


class JevUnavailable(RuntimeError):
    """Raised when the Jev endpoint cannot provide a valid decision."""


@dataclass(frozen=True)
class JevQuestion:
    """One typed question: ``noul`` (yes/no), ``choice``, or ``score``."""

    name: str
    kind: str
    instructions: str
    criteria: Any = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.kind, "instructions": self.instructions}
        if self.criteria is not None:
            payload["criteria"] = self.criteria
        return payload


@dataclass(frozen=True)
class JevAnswer:
    name: str
    kind: str
    value: float | str
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass(frozen=True)
class JevResponse:
    answers: dict[str, JevAnswer]
    model: str
    elapsed_seconds: float


def _clamped(value: Any, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _hash_prefix(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:10]


class JevClient:
    """Batch-question client. Every failure surfaces as :class:`JevUnavailable`.

    Modes:
      - ``live``: real HTTP calls to ``base_url``; API key from ``api_key``
        argument or the ``api_key_env`` environment variable.
      - ``mock``: deterministic answers from ``mock_answers``; no network.
    """

    def __init__(
        self,
        *,
        mode: str = "live",
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "",
        api_key_env: str = DEFAULT_API_KEY_ENV,
        timeout_seconds: float = 2.0,
        mock_answers: Mapping[str, Any] | None = None,
        retries: int = 1,
    ) -> None:
        if mode not in {"live", "mock"}:
            raise ValueError("mode must be 'live' or 'mock'")
        self.mode = mode
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.mock_answers = dict(mock_answers or {})
        self.retries = max(0, int(retries))
        self.api_key = str(api_key or "")
        if self.mode == "live":
            if not self.api_key and api_key_env:
                self.api_key = str(os.environ.get(api_key_env) or "")
            if not self.api_key:
                raise JevUnavailable(f"missing API key (env var: {api_key_env or 'none'})")

    @classmethod
    def from_env(cls, **overrides: Any) -> "JevClient":
        """Build a live client reading TYPESAFE_API_KEY from the environment."""
        return cls(mode="live", **overrides)

    def ask(self, state: Any, questions: Sequence[JevQuestion]) -> JevResponse:
        if self.mode == "mock":
            return self._mock_response(questions)
        state_payload: Any = (
            json.dumps(state, ensure_ascii=False) if isinstance(state, Mapping) else str(state)
        )
        payload = {
            "model": self.model,
            "state": state_payload,
            "questions": {q.name: q.to_payload() for q in questions},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        started = time.perf_counter()
        last_error: Exception | None = None
        for _attempt in range(self.retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(
                        f"{self.base_url}{ENDPOINT_PATH}", json=payload, headers=headers
                    )
                if response.status_code >= 500 or response.status_code == 429:
                    raise JevUnavailable(f"transient status {response.status_code}")
                response.raise_for_status()
                return self._parse_response(response.json(), started)
            except (httpx.HTTPError, JevUnavailable, ValueError, TypeError, KeyError) as exc:
                last_error = exc
        raise JevUnavailable(str(last_error or "jev request failed"))

    def _parse_response(self, data: Any, started: float) -> JevResponse:
        if not isinstance(data, dict):
            raise ValueError("jev response must be an object")
        raw_answers = data.get("answers")
        if not isinstance(raw_answers, dict):
            raise ValueError("jev response missing answers object")
        answers: dict[str, JevAnswer] = {}
        for name, raw in raw_answers.items():
            if not isinstance(raw, dict):
                continue
            kind = str(raw.get("type") or "")
            if kind == "noul":
                value: float | str = float(raw.get("noul", 0.0))
                confidence = float(value)
            elif kind == "choice":
                value = str(raw.get("choice") or "")
                confidence = _clamped(raw.get("confidence"), 0.0)
            elif kind == "score":
                value = float(raw.get("score", 0.0))
                confidence = _clamped(raw.get("confidence"), 0.0)
            else:
                continue
            probabilities_raw = raw.get("probabilities")
            probabilities_raw = probabilities_raw if isinstance(probabilities_raw, dict) else {}
            probabilities = {
                str(k): float(v)
                for k, v in probabilities_raw.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            answers[str(name)] = JevAnswer(
                name=str(name), kind=kind, value=value, probabilities=probabilities, confidence=confidence
            )
        return JevResponse(
            answers=answers,
            model=str(data.get("model") or self.model),
            elapsed_seconds=time.perf_counter() - started,
        )

    def _mock_response(self, questions: Sequence[JevQuestion]) -> JevResponse:
        answers: dict[str, JevAnswer] = {}
        for question in questions:
            configured = self.mock_answers.get(question.name)
            configured = configured if isinstance(configured, dict) else {}
            if question.kind == "noul":
                value = _clamped(configured.get("noul"), 0.0)
                answers[question.name] = JevAnswer(question.name, "noul", value, confidence=value)
            elif question.kind == "choice":
                answers[question.name] = JevAnswer(
                    question.name,
                    "choice",
                    str(configured.get("choice") or ""),
                    confidence=_clamped(configured.get("confidence"), 0.0),
                )
            elif question.kind == "score":
                try:
                    value = float(configured.get("score", 0.0))
                except (TypeError, ValueError):
                    value = 0.0
                answers[question.name] = JevAnswer(
                    question.name, "score", value, confidence=_clamped(configured.get("confidence"), 0.0)
                )
        return JevResponse(answers=answers, model=f"{self.model}+mock", elapsed_seconds=0.0)


def log_decision(
    access_point: str,
    *,
    state: Any,
    questions: Sequence[JevQuestion],
    response: JevResponse | None = None,
    adopted: bool,
    elapsed: float,
    cache_hit: bool = False,
    error: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Emit one bounded, desensitized decision-log line (no raw content)."""
    state_text = json.dumps(state, ensure_ascii=False) if isinstance(state, Mapping) else str(state)
    answers = getattr(response, "answers", None) or {}
    question_logs: list[dict[str, Any]] = []
    for question in questions:
        answer = answers.get(question.name)
        entry: dict[str, Any] = {"name": question.name, "kind": question.kind}
        if answer is not None:
            entry["value"] = answer.value
            entry["confidence"] = round(answer.confidence, 4)
        question_logs.append(entry)
    payload: dict[str, Any] = {
        "access_point": access_point,
        "adopted": bool(adopted),
        "model": str(getattr(response, "model", "") or ""),
        "elapsed_seconds": round(float(elapsed), 4),
        "cache_hit": bool(cache_hit),
        "state_chars": len(state_text),
        "state_hash": _hash_prefix(state_text),
        "questions": question_logs,
        "error": error,
    }
    if extra:
        payload["extra"] = dict(extra)
    LOGGER.info("jev_decision %s", json.dumps(payload, ensure_ascii=False))
