"""
qwen25_coder_server.py - Kaggle Bootstrap & Supervisor for Qwen 2.5 Coder 14B.

Serves Qwen2.5-Coder-14B-Instruct-GGUF (Q4_K_M, ~9.0 GB) on 2x Tesla T4 GPUs.
Features:
- Instant Time-to-First-Token (TTFT < 1s)
- Pure coding instruction mode (0s deliberation delay)
- High-throughput inference (~30-35 tokens/sec)
- 8192 context window (fits easily into 32 GB VRAM)
- Automated 4-hour Cloudflare Quick Tunnel watchdog & ntfy.sh broadcast
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request

WORKING_DIR = Path("/kaggle/working")
MODEL_DIR = WORKING_DIR / "qwen-model"
MODEL_PATH = MODEL_DIR / "qwen2.5-coder-14b-instruct-q4_k_m.gguf"

LLAMA_SERVER_BIN = WORKING_DIR / "llama.cpp" / "build" / "bin" / "llama-server"
CUDA_LIBS_DIR = WORKING_DIR / "cuda-libs"
CLOUDFLARED_BIN = WORKING_DIR / "cloudflared"

LLAMA_LOG_PATH = WORKING_DIR / "llama-server.log"
CLOUDFLARED_LOG_PATH = WORKING_DIR / "cloudflared.log"
LLAMA_PID_FILE = WORKING_DIR / "llama_server.pid"
CLOUDFLARED_PID_FILE = WORKING_DIR / "cloudflared.pid"

LOCAL_PORT = 8000
HEALTH_URL = f"http://127.0.0.1:{LOCAL_PORT}/health"
DEFAULT_API_KEY = os.getenv("QWEN_API_KEY", "my_secret")


def log(msg: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def check_gpu() -> None:
    log("Checking GPU status...")
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
                text=True,
            )
            for line in out.strip().splitlines():
                log(f"GPU: {line.strip()}")
        except Exception as exc:
            log(f"nvidia-smi query failed: {exc}")
    else:
        log("WARNING: nvidia-smi not found! Running on CPU?")


def ensure_cloudflared() -> Path:
    if CLOUDFLARED_BIN.exists() and os.access(CLOUDFLARED_BIN, os.X_OK):
        return CLOUDFLARED_BIN

    log(f"cloudflared not found at {CLOUDFLARED_BIN}. Downloading latest release...")
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    try:
        req = Request(url, headers={"User-Agent": "curl/7.68.0"})
        with urlopen(req, timeout=60) as resp, open(CLOUDFLARED_BIN, "wb") as out:
            shutil.copyfileobj(resp, out)
        os.chmod(CLOUDFLARED_BIN, 0o755)
        log("Successfully downloaded and installed cloudflared.")
    except Exception as exc:
        log(f"ERROR: Failed to download cloudflared: {exc}")
        raise
    return CLOUDFLARED_BIN


def ensure_model_and_bins() -> None:
    """Ensure Qwen 2.5 Coder 14B model, llama-server, and CUDA libs are ready."""
    log("Verifying prerequisites for Qwen 2.5 Coder 14B...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Model resolution
    if not MODEL_PATH.exists():
        # Check /kaggle/input for any mounted dataset
        input_dir = Path("/kaggle/input")
        candidates = list(input_dir.glob("*/*qwen2*5*coder*14b*.gguf")) + list(input_dir.glob("*/*Qwen2.5-Coder-14B*.gguf"))
        if candidates:
            log(f"Found model in persistent dataset: {candidates[0]}")
            MODEL_PATH.symlink_to(candidates[0])
        else:
            log("Model not found in /kaggle/input. Downloading Qwen2.5-Coder-14B-Instruct-GGUF from Hugging Face (~90s)...")
            dl_cmd = [
                "huggingface-cli", "download",
                "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
                "qwen2.5-coder-14b-instruct-q4_k_m.gguf",
                "--local-dir", str(MODEL_DIR),
                "--local-dir-use-symlinks", "False",
            ]
            subprocess.run(dl_cmd, check=True)

    size_gb = MODEL_PATH.stat().st_size / (1024 ** 3)
    log(f"Model confirmed: {MODEL_PATH} ({size_gb:.2f} GB)")

    # 2. Binary and CUDA libraries setup from persistent Kaggle dataset
    if not LLAMA_SERVER_BIN.exists():
        server_dataset = Path("/kaggle/input/llama-server-cuda-t4")
        if server_dataset.exists():
            log(f"Setting up llama-server and CUDA libs from {server_dataset}...")
            CUDA_LIBS_DIR.mkdir(parents=True, exist_ok=True)
            LLAMA_SERVER_BIN.parent.mkdir(parents=True, exist_ok=True)
            for item in server_dataset.rglob("*"):
                if item.is_file():
                    shutil.copy2(item, CUDA_LIBS_DIR / item.name)
                    shutil.copy2(item, LLAMA_SERVER_BIN.parent / item.name)

            server_bin_source = CUDA_LIBS_DIR / "llama-server"
            if server_bin_source.exists():
                os.chmod(server_bin_source, 0o755)
                os.chmod(LLAMA_SERVER_BIN, 0o755)

    if not LLAMA_SERVER_BIN.exists():
        raise FileNotFoundError(
            f"llama-server binary not found at {LLAMA_SERVER_BIN}.\n"
            "Please ensure dataset 'shivakrishna55/llama-server-cuda-t4' is attached to this notebook."
        )
    log(f"llama-server binary confirmed: {LLAMA_SERVER_BIN}")

    # 3. Ensure native libmtmd.so is built if missing (~25s)
    mtmd_target = LLAMA_SERVER_BIN.parent / "libmtmd.so.0"
    if not mtmd_target.exists() or mtmd_target.stat().st_size < 100000:
        log("Building native libmtmd.so from llama.cpp source (~30s)...")
        try:
            src_dir = WORKING_DIR / "llama-src"
            if not src_dir.exists():
                subprocess.run(
                    ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(src_dir)],
                    check=True,
                )
            subprocess.run(
                ["cmake", "-B", str(src_dir / "build"), "-DGGML_CUDA=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF"],
                cwd=str(src_dir),
                check=True,
            )
            subprocess.run(
                ["cmake", "--build", str(src_dir / "build"), "--target", "mtmd", "-j4"],
                cwd=str(src_dir),
                check=True,
            )
            for name in ["libmtmd.so", "libmtmd.so.0", "libmtmd.so.0.4.1"]:
                built_file = src_dir / "build" / "bin" / name
                if built_file.exists():
                    shutil.copy2(built_file, LLAMA_SERVER_BIN.parent / name)
                    shutil.copy2(built_file, CUDA_LIBS_DIR / name)
                    shutil.copy2(built_file, WORKING_DIR / name)
            log("Native libmtmd.so built and installed successfully.")
        except Exception as exc:
            log(f"WARNING: Could not build libmtmd.so: {exc}")

    ensure_cloudflared()


def is_local_health_ok(api_key: str = "") -> bool:
    try:
        headers = {"User-Agent": "HealthChecker/1.0"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = Request(HEALTH_URL, headers=headers)
        with urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def get_stored_pid(pid_file: Path) -> int | None:
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if is_process_running(pid):
                return pid
        except Exception:
            pass
    return None


def start_llama_server(api_key: str) -> subprocess.Popen:
    """Start llama-server with Qwen 2.5 Coder 14B."""
    existing_pid = get_stored_pid(LLAMA_PID_FILE)
    if existing_pid and is_local_health_ok(api_key):
        log(f"llama-server is already running with PID {existing_pid} and passing health checks.")
        return None

    log("Starting llama-server for Qwen 2.5 Coder 14B...")
    env = os.environ.copy()
    ld_paths = [
        str(LLAMA_SERVER_BIN.parent),
        str(CUDA_LIBS_DIR),
        str(WORKING_DIR),
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
    ]
    env["LD_LIBRARY_PATH"] = ":".join(ld_paths) + ":" + env.get("LD_LIBRARY_PATH", "")

    # Qwen 2.5 Coder 14B flags: 8192 context, clean tensor-split, NO flash-attn bug
    cmd = [
        str(LLAMA_SERVER_BIN),
        "-m", str(MODEL_PATH),
        "--host", "0.0.0.0",
        "--port", str(LOCAL_PORT),
        "-ngl", "99",
        "--tensor-split", "1,1",
        "-c", "8192",
        "--parallel", "1",
        "--api-key", api_key,
    ]

    log_file = open(LLAMA_LOG_PATH, "a")
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )

    LLAMA_PID_FILE.write_text(str(process.pid))
    log(f"llama-server spawned with PID {process.pid}. Waiting for model initialization...")

    max_wait = 180  # 14B model loads fast (~30-45s)
    start_time = time.time()
    last_log_time = start_time

    while time.time() - start_time < max_wait:
        if process.poll() is not None:
            err_log = LLAMA_LOG_PATH.read_text(errors="ignore") if LLAMA_LOG_PATH.exists() else "No log"
            log(f"llama-server log output on exit:\n{err_log}")
            raise RuntimeError(f"llama-server exited prematurely with code {process.returncode}!")
        if is_local_health_ok(api_key):
            log(f"llama-server is HEALTHY (200 OK) after {time.time() - start_time:.1f}s.")
            return process

        if time.time() - last_log_time >= 15:
            last_log_time = time.time()
            elapsed = int(time.time() - start_time)
            log(f"[{elapsed}s] Loading Qwen 2.5 Coder 14B weights into 2x T4 VRAM...")

        time.sleep(2)

    raise TimeoutError("Timed out waiting for llama-server /health after 180s.")


def extract_tunnel_url(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    content = log_path.read_text(errors="ignore")
    matches = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
    if matches:
        return matches[-1]
    return None


def broadcast_tunnel_url(url: str) -> None:
    """Broadcast current URL to ntfy.sh for instant Windows client discovery."""
    topic = os.getenv("QWEN_SYNC_TOPIC", "kaggle_qwen_shivakrishna55").strip()
    if not topic:
        return
    try:
        req = Request(
            f"https://ntfy.sh/{topic}",
            data=f"CLOUDFLARE_TUNNEL_URL={url}".encode("utf-8"),
            headers={"Title": "Kaggle Qwen 2.5 Tunnel Active", "Tags": "rocket,gpu"},
        )
        urlopen(req, timeout=5)
        log(f"Broadcasted tunnel URL to https://ntfy.sh/{topic}")
    except Exception as exc:
        log(f"Warning: ntfy broadcast failed: {exc}")


def start_cloudflared() -> tuple[subprocess.Popen | None, str]:
    existing_pid = get_stored_pid(CLOUDFLARED_PID_FILE)
    existing_url = extract_tunnel_url(CLOUDFLARED_LOG_PATH)

    if existing_pid and existing_url:
        log(f"cloudflared is already running with PID {existing_pid} at {existing_url}")
        return None, existing_url

    if existing_pid:
        log(f"cloudflared process {existing_pid} found but no URL in log. Terminating...")
        try:
            os.kill(existing_pid, signal.SIGKILL)
        except Exception:
            pass

    log("Starting cloudflared Quick Tunnel...")
    if CLOUDFLARED_LOG_PATH.exists():
        try:
            CLOUDFLARED_LOG_PATH.unlink()
        except Exception:
            pass

    cmd = [
        str(CLOUDFLARED_BIN),
        "tunnel",
        "--url", f"http://127.0.0.1:{LOCAL_PORT}",
        "--no-autoupdate",
    ]

    log_file = open(CLOUDFLARED_LOG_PATH, "w")
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )

    CLOUDFLARED_PID_FILE.write_text(str(process.pid))
    log(f"cloudflared spawned with PID {process.pid}. Waiting for public URL...")

    start_time = time.time()
    max_wait = 45
    tunnel_url = None

    while time.time() - start_time < max_wait:
        tunnel_url = extract_tunnel_url(CLOUDFLARED_LOG_PATH)
        if tunnel_url:
            break
        time.sleep(1)

    if not tunnel_url:
        raise TimeoutError("Timed out waiting for Cloudflare Quick Tunnel URL.")

    log(f"CLOUDFLARE QUICK TUNNEL READY: {tunnel_url}")
    broadcast_tunnel_url(tunnel_url)
    return process, tunnel_url


def stop_all_instances() -> None:
    log("Cleaning up existing instances...")
    for pid_file in [LLAMA_PID_FILE, CLOUDFLARED_PID_FILE]:
        pid = get_stored_pid(pid_file)
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                if is_process_running(pid):
                    os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
            if pid_file.exists():
                try:
                    pid_file.unlink()
                except Exception:
                    pass
    subprocess.run(["pkill", "-9", "-f", "llama-server"], check=False)
    subprocess.run(["pkill", "-9", "-f", "cloudflared"], check=False)


def run_supervisor(api_key: str) -> None:
    log("=" * 70)
    log(" KAGGLE QWEN 2.5 CODER 14B SERVER BOOTSTRAP & SUPERVISOR")
    log("=" * 70)

    check_gpu()
    ensure_model_and_bins()
    stop_all_instances()

    llama_proc = start_llama_server(api_key)
    cf_proc, current_url = start_cloudflared()

    print("\n" + "=" * 70)
    print(" ✅ QWEN 2.5 CODER 14B SERVER IS ONLINE AND AUTHENTICATED")
    print("=" * 70)
    print(f" Local Endpoint:       http://127.0.0.1:{LOCAL_PORT}/v1")
    print(f" Public Tunnel URL:    {current_url}")
    print(f" Chat API Endpoint:    {current_url}/v1/chat/completions")
    print(f" Expected Speed:       ~30-35 tokens/sec (0s deliberation delay)")
    print(f" Context Window:       8192 tokens")
    print("=" * 70 + "\n")

    check_interval = 60
    tunnel_start_time = time.time()
    tunnel_max_lifetime = 3.8 * 3600  # 3.8 hours

    while True:
        try:
            time.sleep(check_interval)

            if not is_local_health_ok(api_key):
                log("WARNING: llama-server failed health check! Attempting restart...")
                stop_all_instances()
                llama_proc = start_llama_server(api_key)
                cf_proc, current_url = start_cloudflared()
                continue

            tunnel_age = time.time() - tunnel_start_time
            if tunnel_age > tunnel_max_lifetime:
                log(f"Tunnel age ({tunnel_age/3600:.1f}h) approaching 4h limit. Rotating tunnel...")
                cf_pid = get_stored_pid(CLOUDFLARED_PID_FILE)
                if cf_pid:
                    try:
                        os.kill(cf_pid, signal.SIGKILL)
                    except Exception:
                        pass
                cf_proc, current_url = start_cloudflared()
                tunnel_start_time = time.time()

        except KeyboardInterrupt:
            log("Shutdown requested by user. Cleaning up...")
            stop_all_instances()
            break
        except Exception as exc:
            log(f"Supervisor error: {exc}")
            time.sleep(10)


if __name__ == "__main__":
    api_key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_API_KEY
    run_supervisor(api_key)
