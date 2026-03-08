#!/usr/bin/env python3
"""Simple stdin/stdout chat client for llama-server chat completions."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    messages: list[dict[str, str]] = []
    print(f"[chat-runtime] model={args.model} port={args.port}")
    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": prompt})
        payload = json.dumps(
            {
                "model": args.model,
                "messages": messages,
                "max_tokens": 512,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices", []) if isinstance(body, dict) else []
        content = ""
        if choices and isinstance(choices[0], dict):
            content = str(choices[0].get("message", {}).get("content", "")).strip()
        print(content)
        messages.append({"role": "assistant", "content": content})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
