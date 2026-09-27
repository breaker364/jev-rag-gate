"""Client tests against a local HTTP stub server (no external network)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from jev_rag_gate import JevClient, JevQuestion, JevUnavailable


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.server.requests.append(
            {
                "path": self.path,
                "auth": self.headers.get("Authorization", ""),
                "body": json.loads(self.rfile.read(length) or b"{}"),
            }
        )
        status, payload = self.server.responses.pop(0)
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence
        return


@pytest.fixture()
def jev_server(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    server.requests = []
    server.responses = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-123")
    yield server
    server.shutdown()
    server.server_close()


def _client(server, **overrides):
    return JevClient(
        mode="live",
        base_url=f"http://127.0.0.1:{server.server_port}",
        timeout_seconds=2.0,
        **overrides,
    )


def test_request_shape_and_auth(jev_server):
    jev_server.responses.append(
        (
            200,
            {
                "model": "jev-1.13.0",
                "answers": {"worth": {"type": "noul", "noul": 0.93}},
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )
    )
    response = _client(jev_server).ask(
        "some user message",
        [JevQuestion("worth", "noul", "Is this worth keeping?")],
    )
    assert response.answers["worth"].value == pytest.approx(0.93)
    assert response.model == "jev-1.13.0"
    request = jev_server.requests[0]
    assert request["path"] == "/v1/systemone"
    assert request["auth"] == "Bearer test-key-123"
    assert request["body"]["model"] == "jev-1.13.0"
    assert request["body"]["state"] == "some user message"
    assert request["body"]["questions"]["worth"]["type"] == "noul"


def test_explicit_api_key_beats_env(jev_server, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")
    jev_server.responses.append(
        (200, {"model": "m", "answers": {"ok": {"type": "noul", "noul": 1.0}}})
    )
    _client(jev_server, api_key="explicit-key").ask("s", [JevQuestion("ok", "noul", "ok?")])
    assert jev_server.requests[0]["auth"] == "Bearer explicit-key"


def test_retries_once_on_server_error(jev_server):
    jev_server.responses.append((500, {"error": "boom"}))
    jev_server.responses.append(
        (200, {"model": "m", "answers": {"ok": {"type": "noul", "noul": 0.5}}})
    )
    response = _client(jev_server).ask("state", [JevQuestion("ok", "noul", "ok?")])
    assert response.answers["ok"].value == pytest.approx(0.5)
    assert len(jev_server.requests) == 2


def test_unavailable_after_retry(jev_server):
    jev_server.responses.append((500, {"error": "boom"}))
    jev_server.responses.append((500, {"error": "boom"}))
    with pytest.raises(JevUnavailable):
        _client(jev_server).ask("state", [JevQuestion("ok", "noul", "ok?")])
    assert len(jev_server.requests) == 2


def test_unreachable_raises_unavailable():
    client = JevClient(mode="live", base_url="http://127.0.0.1:9", timeout_seconds=0.5, api_key="k")
    with pytest.raises(JevUnavailable):
        client.ask("state", [JevQuestion("ok", "noul", "ok?")])


def test_missing_key_raises():
    with pytest.raises(JevUnavailable):
        JevClient(mode="live", api_key_env="DEFINITELY_NOT_SET_ENV_VAR_XYZ")


def test_mock_mode_deterministic_without_network():
    client = JevClient(
        mode="mock",
        mock_answers={
            "worth": {"noul": 0.9},
            "kind": {"choice": "execute", "confidence": 0.8},
        },
    )
    questions = [
        JevQuestion("worth", "noul", "worth remembering"),
        JevQuestion("kind", "choice", "pick a kind", criteria={"execute": "do", "research": "look"}),
    ]
    first = client.ask("state text", questions)
    second = client.ask("state text", questions)
    assert first.answers["worth"].value == pytest.approx(0.9)
    assert first.answers["kind"].value == "execute"
    assert first == second
