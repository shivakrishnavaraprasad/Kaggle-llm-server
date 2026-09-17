"""
update_windows_env.py - Safely update the Cloudflare Quick Tunnel URL in .env.

Usage:
    python update_windows_env.py <https://your-tunnel.trycloudflare.com>
    python update_windows_env.py --url <https://your-tunnel.trycloudflare.com>
    python update_windows_env.py  (will prompt interactively)
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

ENV_PATH = Path(__file__).resolve().parent / ".env"
REQUIRED_SCHEME = "https"


def validate_url(url: str) -> str:
    """Validate and normalize the provided Cloudflare tunnel URL."""
    clean_url = url.strip().rstrip("/")
    if not clean_url:
        raise ValueError("URL cannot be empty.")

    parsed = urlparse(clean_url)
    if parsed.scheme.lower() != REQUIRED_SCHEME:
        raise ValueError(
            f"Invalid URL scheme '{parsed.scheme}'. Must be '{REQUIRED_SCHEME}' (e.g., https://xxx.trycloudflare.com)"
        )

    if not parsed.netloc:
        raise ValueError(f"Invalid host in URL: {clean_url}")

    # Remove any path suffix like /v1 or /health if accidentally supplied
    normalized_url = f"{parsed.scheme}://{parsed.netloc}"
    return normalized_url


def backup_env_file(env_file: Path) -> Path | None:
    """Create a timestamped backup of .env if it exists."""
    if not env_file.exists():
        return None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = env_file.parent / f".env.bak.{timestamp}"
    shutil.copy2(env_file, backup_file)
    return backup_file


def update_env_url(new_url: str, env_file: Path = ENV_PATH) -> bool:
    """
    Update or insert QWEN_BASE_URL in the .env file.
    Preserves all comments, existing variables, and never exposes QWEN_API_KEY.
    """
    validated_url = validate_url(new_url)

    # Backup existing file before any modification
    backup_path = backup_env_file(env_file)
    if backup_path:
        print(f"[+] Backup created: {backup_path.name}")

    lines: list[str] = []
    if env_file.exists():
        lines = env_file.read_text(encoding="utf-8").splitlines()

    updated = False
    new_lines: list[str] = []
    key_pattern = re.compile(r"^\s*QWEN_BASE_URL\s*=")

    for line in lines:
        if key_pattern.match(line):
            # Replace existing key
            new_lines.append(f"QWEN_BASE_URL={validated_url}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        # If QWEN_BASE_URL was not present, append it cleanly
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.append(f"QWEN_BASE_URL={validated_url}")

    # Write back atomically or with proper utf-8 encoding
    content = "\n".join(new_lines) + "\n"
    env_file.write_text(content, encoding="utf-8")

    print(f"[+] Successfully updated QWEN_BASE_URL in {env_file.name}")
    print(f"    New Base URL: {validated_url}")
    print("    Note: Cloudflare Quick Tunnels are ephemeral and change whenever restarted.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely update QWEN_BASE_URL in the Windows .env file."
    )
    parser.add_argument(
        "tunnel_url",
        nargs="?",
        help="The new Cloudflare tunnel URL (e.g. https://xxxx.trycloudflare.com)",
    )
    parser.add_argument(
        "--url",
        dest="flag_url",
        help="Alternative flag to pass the tunnel URL",
    )
    args = parser.parse_args()

    target_url = args.tunnel_url or args.flag_url

    if not target_url:
        try:
            target_url = input("Enter new Cloudflare tunnel URL: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            return 1

    if not target_url:
        print("[-] Error: No URL provided.", file=sys.stderr)
        return 1

    try:
        update_env_url(target_url)
        return 0
    except Exception as exc:
        print(f"[-] Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
