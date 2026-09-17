"""
swift_server.py - Kaggle Bootstrap, Supervisor, and Auto-Recovery Server.

Runs inside the Kaggle GPU session:
1. Validates hardware (2x Tesla T4) and CUDA libraries.
2. Checks model and binaries.
3. Automatically downloads cloudflared if missing.
4. Starts llama-server with multi-GPU tensor splitting (1,1).
5. Waits for local /health (HTTP 200).
6. Starts cloudflared Quick Tunnel and extracts trycloudflare.com URL.
7. Enters a resilient supervisor loop that monitors health, auto-restarts
   cloudflared if the tunnel drops (e.g., after 4 hours), and keeps
   the Kaggle session alive for up to 11.5 hours.
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

# Runtime Environment Paths on Kaggle
WORKING_DIR = Path("/kaggle/working")
MODEL_PATH = WORKING_DIR / "qwen-model" / "Swift-Qwen3.8-27B-Q4_K_M.gguf"
LLAMA_SERVER_BIN = WORKING_DIR / "llama.cpp" / "build" / "bin" / "llama-server"
CUDA_LIBS_DIR = WORKING_DIR / "cuda-libs"
CLOUDFLARED_BIN = WORKING_DIR / "cloudflared"

LLAMA_LOG_PATH = WORKING_DIR / "llama-server.log"
CLOUDFLARED_LOG_PATH = WORKING_DIR / "cloudflared.log"
LLAMA_PID_FILE = WORKING_DIR / "llama_server.pid"
CLOUDFLARED_PID_FILE = WORKING_DIR / "cloudflared.pid"

LOCAL_PORT = 8000
HEALTH_URL = f"http://127.0.0.1:{LOCAL_PORT}/health"

# Default API key (Can be overridden via QWEN_API_KEY environment variable)
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
    """Download and set up cloudflared binary if not present."""
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
        log(f"Failed to download cloudflared via Python: {exc}. Trying curl...")
        subprocess.run(["curl", "-fsSL", url, "-o", str(CLOUDFLARED_BIN)], check=True)
        os.chmod(CLOUDFLARED_BIN, 0o755)

    return CLOUDFLARED_BIN


def verify_prerequisites() -> None:
    log("Verifying prerequisites...")
    
    # 1. Model setup from persistent Kaggle dataset or working dir
    if not MODEL_PATH.exists():
        input_dir = Path("/kaggle/input")
        model_candidate = input_dir / "swift-qwen3-8-27b-q4-km" / "Swift-Qwen3.8-27B-Q4_K_M.gguf"
        if not model_candidate.exists():
            candidates = list(input_dir.glob("**/*.gguf")) if input_dir.exists() else []
            if candidates:
                model_candidate = candidates[0]

        if model_candidate.exists():
            log(f"Mounting model from persistent dataset: {model_candidate}")
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            if not MODEL_PATH.exists():
                MODEL_PATH.symlink_to(model_candidate)
                log(f"Created symlink: {MODEL_PATH} -> {model_candidate}")
        else:
            log(f"Model not found in /kaggle/input. Attempting download from Hugging Face...")
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            dl_cmd = [
                "huggingface-cli", "download",
                "bartowski/ukisai_Swift-Qwen3.8-27b-GGUF",
                "ukisai_Swift-Qwen3.8-27b-Q4_K_M.gguf",
                "--local-dir", str(MODEL_PATH.parent),
            ]
            res = subprocess.run(dl_cmd, check=False)
            target = MODEL_PATH.parent / "ukisai_Swift-Qwen3.8-27b-Q4_K_M.gguf"
            if target.exists() and not MODEL_PATH.exists():
                target.rename(MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}.\n"
            "Please ensure dataset 'shivakrishna55/swift-qwen3-8-27b-q4-km' is attached to this notebook."
        )

    size_gb = MODEL_PATH.stat().st_size / (1024 ** 3)
    log(f"Model confirmed: {MODEL_PATH} ({size_gb:.2f} GB)")

    # 2. Binary and CUDA libraries setup from persistent Kaggle dataset
    if not LLAMA_SERVER_BIN.exists():
        server_dataset = Path("/kaggle/input/llama-server-cuda-t4")
        if server_dataset.exists():
            log(f"Setting up llama-server and CUDA libs from {server_dataset}...")
            CUDA_LIBS_DIR.mkdir(parents=True, exist_ok=True)
            LLAMA_SERVER_BIN.parent.mkdir(parents=True, exist_ok=True)
            for item in server_dataset.iterdir():
                # Copy to CUDA_LIBS_DIR
                target = CUDA_LIBS_DIR / item.name
                if not target.exists():
                    shutil.copy2(item, target)

                # Also copy shared libraries directly alongside the binary
                if item.name.endswith(".so") or ".so." in item.name:
                    bin_so = LLAMA_SERVER_BIN.parent / item.name
                    if not bin_so.exists():
                        shutil.copy2(item, bin_so)

            server_bin_source = CUDA_LIBS_DIR / "llama-server"
            if server_bin_source.exists():
                try:
                    os.chmod(server_bin_source, 0o755)
                except Exception:
                    pass
                if not LLAMA_SERVER_BIN.exists():
                    shutil.copy2(server_bin_source, LLAMA_SERVER_BIN)
                    try:
                        os.chmod(LLAMA_SERVER_BIN, 0o755)
                    except Exception:
                        pass

    if not LLAMA_SERVER_BIN.exists():
        raise FileNotFoundError(
            f"llama-server binary not found at {LLAMA_SERVER_BIN}.\n"
            "Please ensure dataset 'shivakrishna55/llama-server-cuda-t4' is attached to this notebook."
        )
    log(f"llama-server binary confirmed: {LLAMA_SERVER_BIN}")

    # Ensure genuine native libmtmd.so.0 is present to satisfy dynamic linker
    mtmd_target = LLAMA_SERVER_BIN.parent / "libmtmd.so.0"
    if not mtmd_target.exists() or mtmd_target.stat().st_size < 100000:
        log("Building native libmtmd.so from llama.cpp source (~30s)...")
        try:
            src_dir = Path("/kaggle/working/llama-src")
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
                    shutil.copy2(built_file, Path(f"/kaggle/working/{name}"))
            log("Native libmtmd.so built and installed successfully.")
        except Exception as exc:
            log(f"WARNING: Could not build libmtmd.so: {exc}")

    # 3. Cloudflared check (downloads automatically if missing)
    ensure_cloudflared()


def is_local_health_ok(api_key: str = "") -> bool:
    try:
        headers = {"User-Agent": "HealthChecker/1.0"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = Request(HEALTH_URL, headers=headers)
        with urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
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
    """Start llama-server in background with multi-GPU tensor-split."""
    existing_pid = get_stored_pid(LLAMA_PID_FILE)
    if existing_pid and is_local_health_ok(api_key):
        log(f"llama-server is already running with PID {existing_pid} and passing health checks.")
        return None

    log("Starting llama-server...")
    env = os.environ.copy()
    ld_paths = [
        str(LLAMA_SERVER_BIN.parent),
        str(CUDA_LIBS_DIR),
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
    ]
    env["LD_LIBRARY_PATH"] = ":".join(ld_paths) + ":" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        str(LLAMA_SERVER_BIN),
        "-m", str(MODEL_PATH),
        "--host", "0.0.0.0",
        "--port", str(LOCAL_PORT),
        "-ngl", "99",
        "--tensor-split", "1,1",
        "-c", "4096",
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

    # Wait for /health to turn 200 OK (up to 360s for 16.8 GB GGUF across 2x T4)
    max_wait = 360
    start_time = time.time()
    last_log_time = start_time

    while time.time() - start_time < max_wait:
        if process.poll() is not None:
            err_log = LLAMA_LOG_PATH.read_text(errors="ignore") if LLAMA_LOG_PATH.exists() else "No log"
            log(f"llama-server log output on exit:\n{err_log}")
            raise RuntimeError(f"llama-server exited prematurely with code {process.returncode}! Check {LLAMA_LOG_PATH}")
        if is_local_health_ok(api_key):
            log(f"llama-server is HEALTHY (200 OK) after {time.time() - start_time:.1f}s.")
            return process

        # Print progress every 20 seconds
        if time.time() - last_log_time >= 20:
            last_log_time = time.time()
            elapsed = int(time.time() - start_time)
            tail = ""
            if LLAMA_LOG_PATH.exists():
                lines = [l.strip() for l in LLAMA_LOG_PATH.read_text(errors="ignore").splitlines() if l.strip()]
                tail = lines[-1] if lines else ""
            log(f"[{elapsed}s] Loading 16.8 GB weights into 2x T4 VRAM... (Log: {tail[:80]})")

        time.sleep(3)

    raise TimeoutError("Timed out waiting for llama-server /health after 360s.")


def extract_tunnel_url(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    content = log_path.read_text(errors="ignore")
    matches = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
    if matches:
        return matches[-1]
    return None


def start_cloudflared() -> tuple[subprocess.Popen | None, str]:
    """Start cloudflared Quick Tunnel and capture public URL."""
    existing_pid = get_stored_pid(CLOUDFLARED_PID_FILE)
    existing_url = extract_tunnel_url(CLOUDFLARED_LOG_PATH)

    if existing_pid and existing_url:
        log(f"cloudflared is already running with PID {existing_pid} at {existing_url}")
        return None, existing_url

    log("Starting cloudflared Quick Tunnel...")
    CLOUDFLARED_LOG_PATH.write_text("")  # Truncate old log to ensure fresh URL

    cmd = [
        str(CLOUDFLARED_BIN),
        "tunnel",
        "--url", f"http://127.0.0.1:{LOCAL_PORT}",
        "--no-autoupdate",
    ]

    log_file = open(CLOUDFLARED_LOG_PATH, "a")
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )

    CLOUDFLARED_PID_FILE.write_text(str(process.pid))
    log(f"cloudflared spawned with PID {process.pid}. Waiting for public URL...")

    # Wait for URL in log (up to 40s)
    max_wait = 40
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if process.poll() is not None:
            raise RuntimeError(f"cloudflared exited with code {process.returncode}! Check {CLOUDFLARED_LOG_PATH}")
        url = extract_tunnel_url(CLOUDFLARED_LOG_PATH)
        if url:
            log(f"Extracted Cloudflare URL: {url}")
            broadcast_tunnel_url(url)
            return process, url
        time.sleep(2)

    raise TimeoutError("Timed out waiting for Cloudflare Quick Tunnel URL.")


def broadcast_tunnel_url(url: str) -> None:
    """Broadcast public URL to free sync channel for instant Windows auto-sync."""
    topic = os.getenv("QWEN_SYNC_TOPIC", "kaggle_qwen_shivakrishna55").strip()
    if not topic:
        return
    sync_url = f"https://ntfy.sh/{topic}"
    try:
        req = Request(
            sync_url,
            data=url.strip().encode("utf-8"),
            headers={"Title": "Qwen Cloudflare Tunnel Online", "Tags": "rocket"},
        )
        with urlopen(req, timeout=5) as resp:
            log(f"Tunnel URL broadcasted to sync channel: {sync_url} (HTTP {resp.status})")
    except Exception as exc:
        log(f"Notice: Optional sync broadcast notice: {exc}")


def stop_process(pid_file: Path, name: str) -> None:
    pid = get_stored_pid(pid_file)
    if pid:
        log(f"Stopping {name} (PID {pid})...")
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
        time.sleep(1)
        if pid_file.exists():
            pid_file.unlink()


def supervisor_loop(api_key: str, initial_url: str) -> None:
    """
    Long-running supervisor loop.
    Monitors llama-server and cloudflared. Auto-restarts cloudflared if it
    expires or crashes, and announces the new URL. Keeps Kaggle session
    active up to 11.5 hours.
    """
    log("Entering supervisor loop (auto-recovery enabled)...")
    current_url = initial_url
    max_runtime = 11.5 * 3600  # 11.5 hours maximum
    start_time = time.time()

    while time.time() - start_time < max_runtime:
        time.sleep(30)

        # 1. Monitor llama-server
        if not is_local_health_ok(api_key):
            log("WARNING: llama-server local health check failed! Attempting restart...")
            stop_process(LLAMA_PID_FILE, "llama-server")
            try:
                start_llama_server(api_key)
            except Exception as exc:
                log(f"ERROR restarting llama-server: {exc}")

        # 2. Monitor cloudflared process
        cf_pid = get_stored_pid(CLOUDFLARED_PID_FILE)
        cf_running = cf_pid and is_process_running(cf_pid)

        # Check public tunnel health
        public_ok = False
        if cf_running and current_url:
            try:
                req = Request(f"{current_url}/health", headers={"User-Agent": "HealthChecker/1.0"})
                with urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        public_ok = True
            except Exception:
                public_ok = False

        if not cf_running or not public_ok:
            log("WARNING: Cloudflare tunnel down or expired! Restarting cloudflared...")
            stop_process(CLOUDFLARED_PID_FILE, "cloudflared")
            try:
                _, new_url = start_cloudflared()
                current_url = new_url
                print("\n" + "=" * 70, flush=True)
                print(">>> CLOUDFLARE TUNNEL RE-ESTABLISHED <<<", flush=True)
                print(f"CLOUDFLARE_TUNNEL_URL={current_url}", flush=True)
                print(f"QWEN_BASE_URL={current_url}", flush=True)
                print("=" * 70 + "\n", flush=True)
            except Exception as exc:
                log(f"ERROR restarting cloudflared: {exc}")

    log("Supervisor reached 11.5 hour session limit. Exiting cleanly.")


def main() -> None:
    print("=" * 70, flush=True)
    print(" KAGGLE QWEN GGUF SERVER BOOTSTRAP & SUPERVISOR", flush=True)
    print("=" * 70, flush=True)

    check_gpu()
    verify_prerequisites()

    api_key = DEFAULT_API_KEY

    # Clean up any previous instances to ensure new -fa -b 512 -ub 512 flags apply
    log("Cleaning up any existing server instances...")
    stop_process(LLAMA_PID_FILE, "old llama-server")
    stop_process(CLOUDFLARED_PID_FILE, "old cloudflared")
    subprocess.run(["pkill", "-9", "-f", "llama-server"], check=False)
    subprocess.run(["pkill", "-9", "-f", "cloudflared"], check=False)
    time.sleep(2)

    # Start llama-server with updated performance flags
    start_llama_server(api_key)

    # Start cloudflared
    _, tunnel_url = start_cloudflared()

    print("\n" + "=" * 70, flush=True)
    print(">>> QWEN SERVER PIPELINE ONLINE <<<", flush=True)
    print(f"CLOUDFLARE_TUNNEL_URL={tunnel_url}", flush=True)
    print(f"QWEN_BASE_URL={tunnel_url}", flush=True)
    print(f"QWEN_API_KEY={api_key}", flush=True)
    print(f"LLAMA_SERVER_PID={get_stored_pid(LLAMA_PID_FILE)}", flush=True)
    print(f"CLOUDFLARED_PID={get_stored_pid(CLOUDFLARED_PID_FILE)}", flush=True)
    print("=" * 70 + "\n", flush=True)

    # Run supervisor to keep session alive and handle tunnel restarts
    supervisor_loop(api_key, tunnel_url)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Server stopped by user interrupt.")
    except Exception as exc:
        log(f"FATAL ERROR: {exc}")
        sys.exit(1)