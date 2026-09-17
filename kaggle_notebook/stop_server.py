"""
stop_server.py - Clean shutdown utility for Kaggle llama-server and cloudflared.

Usage:
    python stop_server.py
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

WORKING_DIR = Path("/kaggle/working")
LLAMA_PID_FILE = WORKING_DIR / "llama_server.pid"
CLOUDFLARED_PID_FILE = WORKING_DIR / "cloudflared.pid"


def kill_pid_file(pid_file: Path, name: str) -> None:
    if not pid_file.exists():
        print(f"[-] No PID file found for {name} ({pid_file})")
        return

    try:
        pid = int(pid_file.read_text().strip())
        print(f"[*] Stopping {name} (PID {pid})...")
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
        time.sleep(1)
        pid_file.unlink(missing_ok=True)
        print(f"[+] {name} stopped successfully.")
    except Exception as exc:
        print(f"[-] Failed to stop {name}: {exc}")


def pkill_fallback(pattern: str) -> None:
    try:
        subprocess.run(["pkill", "-9", "-f", pattern], check=False)
    except Exception:
        pass


def main() -> None:
    print("=" * 60)
    print(" STOPPING KAGGLE QWEN SERVER SERVICES")
    print("=" * 60)

    kill_pid_file(CLOUDFLARED_PID_FILE, "cloudflared")
    kill_pid_file(LLAMA_PID_FILE, "llama-server")

    # Fallback cleanup
    pkill_fallback("cloudflared")
    pkill_fallback("llama-server")

    print("[+] All services halted cleanly.")


if __name__ == "__main__":
    main()
