"""
openai_compatible_client.py - Minimal OpenAI SDK Client Example.

Usage:
    python openai_compatible_client.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

raw_base_url = os.getenv("QWEN_BASE_URL", "").strip().rstrip("/")
api_key = os.getenv("QWEN_API_KEY", "").strip()
model = os.getenv("QWEN_MODEL", "qwen").strip()

if not raw_base_url or not api_key:
    print("[-] Error: Missing QWEN_BASE_URL or QWEN_API_KEY in .env", file=sys.stderr)
    sys.exit(1)

# Ensure base_url ends in /v1 without duplicating /v1/v1
normalized_base_url = raw_base_url[:-3] if raw_base_url.endswith("/v1") else raw_base_url
api_endpoint = f"{normalized_base_url}/v1"

try:
    from openai import OpenAI
except ImportError:
    print("[-] Error: 'openai' package is not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)

client = OpenAI(
    base_url=api_endpoint,
    api_key=api_key,
    timeout=300.0,
)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Explain Python decorators in 2 concise sentences."
            }
        ],
        temperature=0.7,
        max_tokens=256,
    )

    choice = response.choices[0]
    content = choice.message.content or ""
    reasoning = getattr(choice.message, "reasoning_content", None) or ""

    output = content if content.strip() else f"[Reasoning] {reasoning}"
    print(output)

except Exception as exc:
    print(f"[-] OpenAI API request failed: {exc}", file=sys.stderr)
    sys.exit(1)