"""
test_health.py - Quick connectivity and sanity test for Qwen API on Kaggle.

Usage:
    python test_health.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent / ".env"


def mask_secret(s: str) -> str:
    if not s or len(s) < 8:
        return "***"
    return f"{s[:4]}...{s[-4:]}"


def main() -> int:
    load_dotenv(dotenv_path=ENV_PATH, override=True)

    base_url = os.getenv("QWEN_BASE_URL", "").strip().rstrip("/")
    api_key = os.getenv("QWEN_API_KEY", "").strip()
    model = os.getenv("QWEN_MODEL", "qwen").strip()

    if not base_url or not api_key:
        print("[-] Error: Missing QWEN_BASE_URL or QWEN_API_KEY in .env", file=sys.stderr)
        return 1

    print("=" * 60)
    print(" QUICK HEALTH & SANITY CHECK")
    print("=" * 60)
    print(f"Target URL: {base_url}")
    print(f"API Key:    {mask_secret(api_key)}")
    print(f"Model:      {model}")
    print("-" * 60)

    # 1. Health probe
    health_url = f"{base_url}/health"
    print(f"[*] Probing health endpoint: {health_url}...")
    t0 = time.perf_counter()
    try:
        resp = requests.get(health_url, timeout=15)
        dt = time.perf_counter() - t0
        if resp.status_code == 200:
            print(f"[+] Health Status: HTTP 200 OK ({dt:.2f}s) | Response: {resp.text.strip()}")
        else:
            print(f"[-] Health Status: HTTP {resp.status_code} ({dt:.2f}s) | Body: {resp.text.strip()}")
            return 1
    except requests.exceptions.ConnectionError as exc:
        print(f"[-] Connection Error: Could not reach {health_url}.")
        print("    The Cloudflare tunnel has likely expired or restarted.")
        return 1
    except requests.exceptions.Timeout:
        print(f"[-] Timeout: Health check timed out after 15s.")
        return 1
    except Exception as exc:
        print(f"[-] Unexpected error: {exc}")
        return 1

    # 2. Short chat test
    chat_url = f"{base_url}/v1/chat/completions"
    print(f"\n[*] Sending test chat completion to: {chat_url}...")
    t0 = time.perf_counter()
    try:
        chat_resp = requests.post(
            chat_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": "Reply with only: Health check confirmed."}
                ],
                "temperature": 0.2,
                "max_tokens": 64,
            },
            timeout=60,
        )
        dt = time.perf_counter() - t0
        chat_resp.raise_for_status()

        data = chat_resp.json()
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "").strip()
        reasoning = msg.get("reasoning_content", "").strip()

        output = content if content else f"[Reasoning] {reasoning}"
        print(f"[+] Chat Response: HTTP 200 OK ({dt:.2f}s)")
        print(f"    Assistant: {output}")

        timings = data.get("timings", {})
        if timings:
            gen_speed = timings.get("predicted_per_second", 0)
            print(f"    Speed:     {gen_speed:.1f} tokens/sec")

        print("\n[+] Health check completed successfully.")
        return 0

    except Exception as exc:
        print(f"[-] Chat completion failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())