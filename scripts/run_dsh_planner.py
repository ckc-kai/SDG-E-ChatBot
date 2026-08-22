#!/usr/bin/env python3
"""Isolated Python 3.10 entrypoint for the DSH planning integration."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig


@contextmanager
def _planner_base_url(base_url: str):
    """Force Ollama's OpenAI-compatible endpoint to honor ``think=false``.

    DSH correctly requests reasoning effort ``off``, but Ollama's compatibility
    endpoint ignores that standard field for Qwen3. A loopback-only adapter
    calls native ``/api/chat`` with ``think=false`` and maps the completed result
    back to the DeepSeek-compatible SSE shape expected by DSH.
    """
    parsed = urlparse(base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.port != 11434:
        yield base_url
        return
    upstream_origin = f"{parsed.scheme}://{parsed.netloc}"

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
            try:
                size = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be an object")
                requested_stream = bool(payload.get("stream"))
                # Ollama's OpenAI-compatible endpoint ignores ``think=false``
                # for Qwen3. Use the native endpoint, which honors it, then map
                # the completed response back to OpenAI/DeepSeek wire format.
                options = {
                    "num_predict": int(payload.get("max_tokens") or 1800),
                    "temperature": float(payload.get("temperature") or 0),
                }
                native_payload = {
                    "model": payload["model"],
                    "messages": payload["messages"],
                    "stream": False,
                    "think": False,
                    "options": options,
                }
                headers = {"Content-Type": "application/json"}
                if authorization := self.headers.get("Authorization"):
                    headers["Authorization"] = authorization
                request = urllib.request.Request(
                    upstream_origin + "/api/chat",
                    data=json.dumps(native_payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=160) as response:
                    native = json.loads(response.read())
                    native_message = native.get("message") or {}
                    completed = {
                        "id": f"chatcmpl-dsh-{int(time.time() * 1000)}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": native.get("model") or payload["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": native_message.get(
                                        "role", "assistant"
                                    ),
                                    "content": native_message.get("content") or "",
                                    **(
                                        {
                                            "reasoning_content": native_message[
                                                "thinking"
                                            ]
                                        }
                                        if native_message.get("thinking")
                                        else {}
                                    ),
                                },
                                "finish_reason": (
                                    "length"
                                    if native.get("done_reason") == "length"
                                    else "stop"
                                ),
                            }
                        ],
                        "usage": {
                            "prompt_tokens": native.get("prompt_eval_count", 0),
                            "completion_tokens": native.get("eval_count", 0),
                            "total_tokens": native.get("prompt_eval_count", 0)
                            + native.get("eval_count", 0),
                        },
                    }
                    if requested_stream:
                        choice = completed["choices"][0]
                        message = choice.get("message") or {}
                        delta = {
                            "role": message.get("role", "assistant"),
                            "content": message.get("content") or "",
                        }
                        reasoning = message.get("reasoning") or message.get(
                            "reasoning_content"
                        )
                        if reasoning:
                            delta["reasoning_content"] = reasoning
                        chunk = {
                            "id": completed.get("id"),
                            "object": "chat.completion.chunk",
                            "created": completed.get("created"),
                            "model": completed.get("model"),
                            "choices": [
                                {
                                    "index": choice.get("index", 0),
                                    "delta": delta,
                                    "finish_reason": choice.get("finish_reason"),
                                }
                            ],
                            "usage": completed.get("usage"),
                        }
                        raw = (
                            "data: "
                            + json.dumps(chunk, separators=(",", ":"))
                            + "\n\ndata: [DONE]\n\n"
                        ).encode("utf-8")
                        content_type = "text/event-stream"
                    else:
                        raw = json.dumps(completed).encode("utf-8")
                        content_type = "application/json"
                    self.send_response(response.status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
            except urllib.error.HTTPError as exc:
                self.send_response(exc.code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(exc.read())
            except Exception as exc:  # return a useful adapter error
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        prompt = payload["prompt"]
        schema = payload["schema"]
        model = str(payload["model"])
        max_tokens = int(payload["max_tokens"])
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(schema, dict):
            raise ValueError("schema must be an object")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"invalid DSH planner input: {exc}", file=sys.stderr)
        return 2

    project_root = Path.cwd().resolve()
    session_root = Path(
        os.environ.get(
            "DSH_SESSION_ROOT",
            str(project_root / "outputs" / "dsh-sessions"),
        )
    ).resolve()
    session_root.mkdir(parents=True, exist_ok=True)
    request = "\n\n".join(
        (
            "/no_think",
            "You are the SDG&E retrieval planner. Do not answer the user's "
            "question and do not inspect workspace files. Produce only one "
            "JSON object, with no Markdown fence or commentary, that satisfies "
            "the supplied JSON Schema.",
            "JSON Schema:\n" + json.dumps(schema, separators=(",", ":")),
            "Planning request:\n" + prompt,
        )
    )
    started = time.perf_counter()
    try:
        base_url = os.environ.get(
            "DSH_BASE_URL", "http://127.0.0.1:11434/v1"
        )
        with _planner_base_url(base_url) as planner_base_url:
            config = DeepSeekHarnessConfig(
                provider=os.environ.get("DSH_PROVIDER", "deepseek-official"),
                model=model,
                max_tokens=max_tokens,
                cwd=str(project_root),
                session_root=str(session_root),
                cordis=os.environ.get("DSH_CORDIS_CONFIG") or None,
                base_url=planner_base_url,
                api_key=os.environ.get("DSH_API_KEY", "ollama"),
                request_timeout_seconds=float(
                    os.environ.get("DSH_RUNTIME_TIMEOUT_SECONDS", "170")
                ),
            )
            with DeepSeekHarness(config) as harness:
                result = harness.run(request)
    except Exception as exc:  # runtime diagnostics are safe and useful to caller
        print(f"DSH runtime error: {exc}", file=sys.stderr)
        return 3
    print(
        json.dumps(
            {
                "output": result.final_response,
                "finish_reason": result.finish_reason,
                "latency_ms": round((time.perf_counter() - started) * 1000),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
