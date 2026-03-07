#!/usr/bin/env python3
"""Thin proxy that fixes Ollama's /v1/completions array-prompt rejection.

lm_eval sends {"prompt": ["str1", "str2", ...]} for loglikelihood tasks.
Ollama's /v1/completions only accepts {"prompt": "string"}.
This proxy sits between them and flattens the array to a joined string.

Usage:
    python3 ollama_completions_proxy.py --port 11435 --ollama-port 11434

Then point lm_eval at http://localhost:11435/v1/completions instead of Ollama.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def flatten_prompt(prompt):
    """Flatten an array prompt into a single string for Ollama."""
    if not isinstance(prompt, list):
        return prompt, False
    parts = []
    for item in prompt:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, list):
            parts.append("".join(str(t) for t in item))
        else:
            parts.append(str(item))
    return "".join(parts), True


def forward_response(handler, resp):
    """Send upstream response back to client."""
    resp_body = resp.read()
    handler.send_response(resp.status)
    for key in ("Content-Type", "Content-Length"):
        val = resp.getheader(key)
        if val:
            handler.send_header(key, val)
    handler.end_headers()
    handler.wfile.write(resp_body)
    return resp.status


def send_error(handler, code, body):
    """Send error response to client."""
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    if isinstance(body, str):
        body = body.encode("utf-8")
    handler.wfile.write(body)


class ProxyHandler(BaseHTTPRequestHandler):
    ollama_base: str = "http://localhost:11434"
    verbose: bool = False

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length else b""
        body = json.loads(raw_body) if raw_body else {}

        prompt = body.get("prompt")
        body["prompt"], flattened = flatten_prompt(prompt)

        target_url = f"{self.ollama_base}{self.path}"
        headers = {"Content-Type": "application/json"}
        auth = self.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth

        req = Request(
            target_url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(req, timeout=300) as resp:
                status = forward_response(self, resp)
                if flattened:
                    model = body.get("model", "?")
                    print(f"[proxy] flattened array prompt -> string | model={model} | {status}")
        except HTTPError as e:
            error_body = e.read()
            send_error(self, e.code, error_body)
            print(f"[proxy] upstream HTTP {e.code}: {error_body[:200]}", file=sys.stderr)
        except (URLError, TimeoutError) as e:
            msg = json.dumps({"error": {"message": f"Proxy upstream error: {e}", "type": "proxy_error"}})
            send_error(self, 502, msg)
            print(f"[proxy] upstream unreachable: {e}", file=sys.stderr)

    def do_GET(self):
        """Pass through GET requests (model listing, health checks)."""
        target_url = f"{self.ollama_base}{self.path}"
        req = Request(target_url, method="GET")
        try:
            with urlopen(req, timeout=30) as resp:
                forward_response(self, resp)
        except HTTPError as e:
            send_error(self, e.code, e.read())
        except (URLError, TimeoutError) as e:
            msg = json.dumps({"error": {"message": f"Proxy upstream error: {e}", "type": "proxy_error"}})
            send_error(self, 502, msg)

    def log_message(self, format, *log_args):
        if self.verbose:
            BaseHTTPRequestHandler.log_message(self, format, *log_args)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Ollama completions proxy — flattens array prompts for lm_eval MC tasks"
    )
    ap.add_argument("--port", type=int, default=11435, help="Port to listen on (default: 11435)")
    ap.add_argument("--ollama-port", type=int, default=11434, help="Ollama port (default: 11434)")
    ap.add_argument("--ollama-host", default="localhost", help="Ollama host (default: localhost)")
    ap.add_argument("--verbose", action="store_true", help="Log every request (noisy)")
    args = ap.parse_args()

    ProxyHandler.ollama_base = f"http://{args.ollama_host}:{args.ollama_port}"
    ProxyHandler.verbose = args.verbose

    server = HTTPServer(("0.0.0.0", args.port), ProxyHandler)
    print(f"[proxy] listening on :{args.port} -> {ProxyHandler.ollama_base}")
    print(f"[proxy] point lm_eval at: http://localhost:{args.port}/v1/completions")
    print(f"[proxy] Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[proxy] shutting down")
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
