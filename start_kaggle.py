"""
start_kaggle.py - Single-Script Lifecycle Manager for Kaggle Qwen GGUF & Cloudflare.

Solves the core operational problems:
1. Validates if the current Cloudflare Quick Tunnel is healthy.
2. If Cloudflare tunnel expired (e.g. after 4h) while Kaggle is still running:
   Extracts the new trycloudflare.com URL from Kaggle logs and auto-updates Windows .env.
3. If Kaggle session expired (after 12h) or stopped:
   Pushes the bootstrap notebook, restarts the session, waits for initialization,
   extracts the new tunnel URL, auto-updates .env, and verifies connectivity.
4. Optional --watch mode: Runs as a background guardian to auto-recover whenever
   the tunnel or session drops.

Usage:
    python start_kaggle.py               # Smart check, auto-update, and auto-restart
    python start_kaggle.py --status      # Check Kaggle kernel status
    python start_kaggle.py --restart     # Force restart Kaggle kernel & update tunnel
    python start_kaggle.py --watch       # Run continuous monitor & auto-recovery
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
ENV_FILE = PROJECT_DIR / ".env"
KAGGLE_DIR = PROJECT_DIR / "kaggle_notebook"
METADATA_FILE = KAGGLE_DIR / "kernel-metadata.json"

# Import update function from update_windows_env
try:
    from update_windows_env import update_env_url
except ImportError:
    def update_env_url(url: str, env_file: Path = ENV_FILE) -> bool:
        # Fallback simple updater
        lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []
        new_lines = [l for l in lines if not re.match(r"^\s*QWEN_BASE_URL\s*=", l)]
        new_lines.append(f"QWEN_BASE_URL={url.strip().rstrip('/')}")
        env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return True


def mask_secret(secret: str | None) -> str:
    if not secret:
        return "<EMPTY>"
    secret = secret.strip()
    if len(secret) <= 8:
        return "***"
    return f"{secret[:4]}...{secret[-4:]}"


def load_environment() -> dict[str, str]:
    if ENV_FILE.exists():
        load_dotenv(dotenv_path=ENV_FILE, override=True)
    else:
        load_dotenv(override=True)

    # Propagate token and username to process environment for kaggle CLI
    token = os.getenv("KAGGLE_API_TOKEN", "").strip()
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token

    username = os.getenv("KAGGLE_USERNAME", "").strip()
    if username:
        os.environ["KAGGLE_USERNAME"] = username

    return {
        "QWEN_BASE_URL": os.getenv("QWEN_BASE_URL", "").strip().rstrip("/"),
        "QWEN_API_KEY": os.getenv("QWEN_API_KEY", "").strip(),
        "KAGGLE_USERNAME": username,
        "KAGGLE_API_TOKEN": token,
        "KAGGLE_KERNEL_SLUG": os.getenv("KAGGLE_KERNEL_SLUG", "").strip(),
    }


def probe_health(base_url: str, timeout: int = 8) -> bool:
    """Check if the given base URL responds with HTTP 200 on /health."""
    if not base_url:
        return False
    try:
        url = f"{base_url.rstrip('/')}/health"
        resp = requests.get(url, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def run_kaggle_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Execute a kaggle CLI command using the local virtual environment Python."""
    venv_py = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
    py_bin = str(venv_py) if venv_py.exists() else sys.executable
    cmd = [py_bin, "-m", "kaggle", "kernels"] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    return result


def get_kernel_slug() -> str:
    slug = os.getenv("KAGGLE_KERNEL_SLUG", "").strip()
    if slug:
        return slug

    if METADATA_FILE.exists():
        try:
            data = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
            return data.get("id", "")
        except Exception:
            pass

    username = os.getenv("KAGGLE_USERNAME", "shivakrishna55")
    return f"{username}/swift-server"


def get_kernel_status(slug: str) -> str:
    res = run_kaggle_command(["status", slug])
    if res.returncode != 0:
        return f"ERROR: {res.stderr.strip() or res.stdout.strip()}"
    return res.stdout.strip()


def extract_urls_from_logs(slug: str) -> list[str]:
    """Extract all trycloudflare.com URLs from the latest kernel execution logs."""
    res = run_kaggle_command(["logs", slug])
    content = res.stdout if res.returncode == 0 else ""
    matches = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
    # Return unique URLs preserving order
    seen = set()
    unique = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


def push_kernel() -> bool:
    """Push code to Kaggle to trigger kernel run."""
    print(f"[*] Pushing notebook folder to Kaggle ({KAGGLE_DIR})...")
    res = run_kaggle_command(["push", "-p", str(KAGGLE_DIR)])
    if res.returncode != 0:
        print(f"[-] Kaggle push failed: {res.stderr.strip() or res.stdout.strip()}", file=sys.stderr)
        return False
    print("[+] Notebook pushed successfully to Kaggle.")
    return True


def get_synced_tunnel_url(timeout: int = 5) -> str | None:
    """Check the broadcast channel for the latest Cloudflare tunnel URL."""
    topic = os.getenv("QWEN_SYNC_TOPIC", "kaggle_qwen_shivakrishna55").strip()
    if not topic:
        return None
    try:
        resp = requests.get(f"https://ntfy.sh/{topic}/raw?poll=1&since=15m", timeout=timeout)
        if resp.status_code == 200:
            urls = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", resp.text)
            if urls:
                return urls[-1]
    except Exception:
        pass
    return None


def poll_for_active_tunnel(slug: str, max_wait: int = 600) -> str | None:
    """Poll Kaggle live log stream and sync channel until an active trycloudflare.com URL appears."""
    print(f"[*] Polling Kaggle kernel status, sync channel, and live logs (max {max_wait}s)...", flush=True)
    start_time = time.time()
    last_status = ""

    while time.time() - start_time < max_wait:
        elapsed = int(time.time() - start_time)

        # 1. Fast check instant sync channel (<0.5s latency)
        sync_url = get_synced_tunnel_url(timeout=3)
        if sync_url:
            print(f"    [{elapsed}s] Detected broadcasted tunnel: {sync_url}", flush=True)
            if probe_health(sync_url, timeout=5):
                print(f"[+] Instant sync verified healthy: {sync_url}", flush=True)
                return sync_url

        # 2. Check kernel status
        status = get_kernel_status(slug)
        if status != last_status:
            print(f"    [{elapsed}s] Kaggle Status: {status}", flush=True)
            last_status = status

        # 3. Check logs if any URL appears
        urls = extract_urls_from_logs(slug)
        if urls:
            latest_url = urls[-1]
            print(f"    [{elapsed}s] Found candidate URL in logs: {latest_url}", flush=True)
            if probe_health(latest_url, timeout=5):
                print(f"[+] Tunnel verified healthy: {latest_url}", flush=True)
                return latest_url

        time.sleep(10)

    return None


def run_pipeline_sync(force_restart: bool = False) -> int:
    """
    Main orchestration routine:
    1. Tests current tunnel. If up & not forcing restart, exits cleanly.
    2. If down or restart forced:
       - Checks Kaggle status.
       - If stopped / 12h expired -> restarts kernel.
       - Polls for new tunnel URL -> updates .env -> runs sanity test.
    """
    env = load_environment()
    base_url = env["QWEN_BASE_URL"]
    api_key = env["QWEN_API_KEY"]
    slug = get_kernel_slug()

    print("=" * 70)
    print(" KAGGLE QWEN GGUF PIPELINE CONTROLLER")
    print("=" * 70)
    print(f"Target Kernel: {slug}")
    print(f"Current Base:  {base_url or '<None>'}")
    print(f"API Key:       {mask_secret(api_key)}")
    print("-" * 70)

    # 1. Health check current URL
    if not force_restart and base_url:
        print(f"[*] Testing current tunnel health: {base_url}/health ...")
        if probe_health(base_url, timeout=5):
            print("\n[✓] CURRENT PIPELINE IS ACTIVE AND HEALTHY!")
            print(f"    Base URL: {base_url}")
            print(f"    API Key:  {mask_secret(api_key)}")
            print("    Run 'python qwen_client.py \"hello\"' or 'python test_health.py' to use it.")
            return 0
        else:
            print("[-] Current tunnel is UNREACHABLE or EXPIRED.")

    # 2. Check Kaggle Kernel Status
    print(f"\n[*] Checking Kaggle kernel status for '{slug}'...")
    status_text = get_kernel_status(slug)
    print(f"    Status: {status_text}")

    # Check if a new URL is already broadcasted or present in logs
    if not force_restart:
        print("[*] Checking broadcast sync channel and Kaggle logs for updated tunnel...")
        sync_candidate = get_synced_tunnel_url(timeout=3)
        if sync_candidate and sync_candidate != base_url:
            print(f"    Found broadcasted URL: {sync_candidate}")
            if probe_health(sync_candidate, timeout=5):
                print(f"[+] Re-connected to fresh tunnel via instant sync: {sync_candidate}")
                update_env_url(sync_candidate, ENV_FILE)
                print("[✓] Pipeline recovered without restarting Kaggle session!")
                return 0

        urls = extract_urls_from_logs(slug)
        if urls and urls[-1] != base_url:
            candidate = urls[-1]
            print(f"    Found recent URL in logs: {candidate}")
            if probe_health(candidate, timeout=5):
                print(f"[+] Re-connected to fresh tunnel: {candidate}")
                update_env_url(candidate, ENV_FILE)
                print("[✓] Pipeline recovered without restarting Kaggle session!")
                return 0

    # 3. If kernel is not running or restart forced, trigger a fresh Kaggle session
    needs_push = force_restart or any(
        kw in status_text.lower()
        for kw in ["complete", "stopped", "error", "canceled", "cancelpending"]
    )

    if needs_push:
        print("\n[*] Kaggle session has ended (12-hour limit) or restart requested.")
        print("[*] Launching fresh Kaggle GPU session...")
        if not push_kernel():
            return 1
        print("[+] Kernel submitted to Kaggle. Waiting for worker to start...")
        time.sleep(15)

    # 4. Poll for the new tunnel URL
    new_url = poll_for_active_tunnel(slug, max_wait=360)
    if not new_url:
        print("[-] Timed out waiting for an active tunnel.", file=sys.stderr)
        print("    Check Kaggle directly at https://www.kaggle.com/code/" + slug, file=sys.stderr)
        return 1

    # 5. Update .env
    update_env_url(new_url, ENV_FILE)

    # 6. Run quick sanity test
    print("\n[*] Running post-sync sanity test...")
    if probe_health(new_url, timeout=10):
        print("\n" + "=" * 70)
        print(" [✓] PIPELINE SYNC COMPLETE & VERIFIED")
        print("=" * 70)
        print(f" New Tunnel URL: {new_url}")
        print(f" Windows .env:   Updated and ready")
        print("=" * 70)
        return 0
    else:
        print("[-] Tunnel URL recorded, but health probe failed.", file=sys.stderr)
        return 1


def watch_loop(interval_seconds: int = 60) -> None:
    """Continuously monitor health and auto-recover if tunnel or session drops."""
    print("=" * 70)
    print(f" QWEN PIPELINE GUARDIAN (Checking every {interval_seconds}s)")
    print(" Press Ctrl+C to stop.")
    print("=" * 70)

    while True:
        try:
            env = load_environment()
            base_url = env["QWEN_BASE_URL"]
            if not probe_health(base_url, timeout=5):
                print(f"\n[!] Tunnel alert at {time.strftime('%X')}: Endpoint unreachable.")
                print("[!] Initiating automatic recovery...")
                run_pipeline_sync(force_restart=False)
            else:
                sys.stdout.write(f"\r[{time.strftime('%X')}] Pipeline OK: {base_url} (probe: 200 OK)   ")
                sys.stdout.flush()

            time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\nGuardian stopped by user.")
            break
        except Exception as exc:
            print(f"\n[-] Watcher error: {exc}")
            time.sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified manager for Kaggle Qwen GGUF and Cloudflare Quick Tunnel."
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check Kaggle kernel status and recent logs.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Force restart the Kaggle GPU session and update tunnel URL.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run continuous guardian mode to automatically detect and fix drops.",
    )
    parser.add_argument(
        "--update-url",
        dest="manual_url",
        help="Manually update QWEN_BASE_URL in .env and test it.",
    )

    args = parser.parse_args()

    if args.manual_url:
        update_env_url(args.manual_url, ENV_FILE)
        sys.exit(0 if probe_health(args.manual_url) else 1)

    if args.status:
        slug = get_kernel_slug()
        print(f"Kernel: {slug}")
        print("Status:", get_kernel_status(slug))
        urls = extract_urls_from_logs(slug)
        print("Tunnel URLs found in logs:", urls)
        sys.exit(0)

    if args.watch:
        watch_loop()
        sys.exit(0)

    # Default: sync pipeline
    code = run_pipeline_sync(force_restart=args.restart)
    sys.exit(code)


if __name__ == "__main__":
    main()