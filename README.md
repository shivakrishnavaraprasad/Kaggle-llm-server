# Kaggle Swift API: Free Self-Hosted Qwen LLM on Kaggle GPUs

Run frontier open-source coding & reasoning models on **2 × NVIDIA Tesla T4 GPUs (32 GB VRAM Total)** on Kaggle for free, expose them through an authenticated **Cloudflare Quick Tunnel**, and connect them directly to your Windows terminal and AI editors (**Cursor, Claude Code, OpenCode, Continue.dev, Aider**).

Includes an automated **single-script lifecycle supervisor** that handles:
- **Cloudflare Quick Tunnel 4-hour expirations** (auto-rotated & synced via `ntfy.sh` without killing your Kaggle session).
- **Kaggle 12-hour session limits** (auto-detects expiration, pushes the notebook, and recovers connectivity).

---

## Model Comparison on 2 × Tesla T4 (32 GB VRAM)

| Model | Size | Context | Speed | Deliberation Delay | Ideal For |
|---|---|---|---|---|---|
| **Swift-Qwen 3.8 27B** *(Distilled Reasoning)* | **16.8 GB** | 4,096 | **~12 tok/s** | ~10–16s | Deep mathematical & architectural reasoning |
| **Qwen 2.5 Coder 14B** *(Recommended for Coding)* | **9.0 GB** | 8,192 | **~32–35 tok/s** | **0s (Instant)** | **Sweet spot**: 3× faster, instant autocompletions & edits |
| **Qwen 2.5 Coder 32B** | 20.0 GB | 4,096 | ~10–12 tok/s | 0s | Maximum code generation accuracy |
| **Qwen 2.5 Coder 7B** | 4.7 GB | 16,384 | ~55–65 tok/s | 0s | Ultra-low latency autocomplete |

---

## Quick Start Guide

### 1. Local Setup on Windows

Clone the repository and install the lightweight dependencies:

```powershell
# 1. Clone repository
git clone https://github.com/<your-username>/kaggle-swift-api.git
cd kaggle-swift-api

# 2. Set up Python virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your local .env file
cp .env.example .env
```

Edit your local `.env` (this file is excluded by `.gitignore` and never committed):
```ini
QWEN_BASE_URL=https://your-active-tunnel.trycloudflare.com
QWEN_API_KEY=my_secret
OPENAI_MODEL=qwen
QWEN_TIMEOUT=300
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_API_TOKEN=KGAT_your_kaggle_token
KAGGLE_KERNEL_SLUG=your_kaggle_username/swift-server
QWEN_SYNC_TOPIC=kaggle_qwen_your_username
```

---

## Running Qwen 3.8 27B on Kaggle (Step-by-Step)

### Step 1: Open Kaggle Notebook
1. Navigate to your Kaggle Notebook: `https://www.kaggle.com/code/<your-username>/swift-server`.
2. Ensure **Accelerator** is set to **GPU T4 x2**.
3. Ensure **Internet** is toggled **ON**.
4. Ensure the following datasets are attached in the right sidebar:
   - `swift-qwen3-8-27b-q4-km` (Contains `Swift-Qwen3.8-27B-Q4_K_M.gguf`, 16.8 GB)
   - `llama-server-cuda-t4` (Contains precompiled CUDA binaries)

### Step 2: Build Native `libmtmd.so` (Runs in ~30s)
The precompiled dataset requires `libmtmd.so`. To ensure 100% GLIBC and ABI compatibility with Kaggle's Ubuntu environment without crashes, compile the lightweight CPU-only target once:

```python
# In a Kaggle notebook cell:
!git clone --depth 1 https://github.com/ggml-org/llama.cpp /kaggle/working/llama-src
!cd /kaggle/working/llama-src && cmake -B build -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF && cmake --build build --target mtmd -j4
!cp -f /kaggle/working/llama-src/build/bin/libmtmd.so* /kaggle/working/llama.cpp/build/bin/
!cp -f /kaggle/working/llama-src/build/bin/libmtmd.so* /kaggle/working/cuda-libs/
!cp -f /kaggle/working/llama-src/build/bin/libmtmd.so* /kaggle/working/
```

### Step 3: Launch `llama-server` & Cloudflare Tunnel
Run the following bootstrap cell:

```python
import os, subprocess, time, requests, re
from pathlib import Path

# Kill any previous processes
!pkill -9 llama-server || true
!pkill -9 cloudflared || true

BIN_DIR = Path("/kaggle/working/llama.cpp/build/bin")
CUDA_DIR = Path("/kaggle/working/cuda-libs")
BIN = BIN_DIR / "llama-server"
MODEL_PATH = Path("/kaggle/working/qwen-model/Swift-Qwen3.8-27B-Q4_K_M.gguf")
CF_BIN = Path("/kaggle/working/cloudflared")

# Download cloudflared if missing
if not CF_BIN.exists():
    !wget -q "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" -O /kaggle/working/cloudflared
    !chmod +x /kaggle/working/cloudflared

API_KEY = "my_secret"
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = f"{BIN_DIR}:{CUDA_DIR}:/kaggle/working:/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")

# Start llama-server on 2x T4 GPUs (Tensor-Split 1,1)
cmd = [
    str(BIN),
    "-m", str(MODEL_PATH),
    "--host", "0.0.0.0",
    "--port", "8000",
    "-ngl", "99",
    "--tensor-split", "1,1",
    "-c", "4096",
    "--parallel", "1",
    "--api-key", API_KEY,
]

log_file = open("/kaggle/working/llama-server.log", "w")
proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
print(f"llama-server spawned with PID {proc.pid}. Loading weights into 2x T4 VRAM (~2 mins)...")

for i in range(120):
    time.sleep(3)
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=2)
        if r.status_code == 200:
            print(f"\n>>> llama-server is HEALTHY (200 OK) after {(i+1)*3}s! <<<")
            break
    except Exception:
        pass
    if i % 7 == 0:
        print(f"Loading weights into 2x T4 VRAM ({(i+1)*3}s)...")

# Launch Cloudflare Quick Tunnel
cf_log = open("/kaggle/working/cloudflared.log", "w")
cf_proc = subprocess.Popen([str(CF_BIN), "tunnel", "--url", "http://127.0.0.1:8000", "--no-autoupdate"], stdout=cf_log, stderr=subprocess.STDOUT)

for _ in range(15):
    time.sleep(2)
    txt = Path("/kaggle/working/cloudflared.log").read_text(errors="ignore")
    m = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", txt)
    if m:
        new_url = m[-1]
        print(f"\nACTIVE CLOUDFLARE URL: {new_url}\n")
        try:
            requests.post("https://ntfy.sh/kaggle_qwen_shivakrishna55", data=f"CLOUDFLARE_TUNNEL_URL={new_url}".encode())
        except Exception:
            pass
        break
```

---

## Windows Client & Terminal CLI

Once the server outputs the active URL, update your Windows `.env`:
```powershell
python update_windows_env.py https://<your-tunnel-url>.trycloudflare.com
```

### 1. Test Health & Latency
```powershell
python test_health.py
```
*Outputs HTTP status, latency, and tokens/second.*

### 2. Dual-Phase Streaming Inference
```powershell
# Interactive chat / code generation:
python qwen_client.py "Write a clean Python function to calculate Fibonacci numbers."

# With reasoning optimization and benchmark metrics:
python qwen_client.py -v --max-tokens 500 --optimize-think "Write an elegant Slack alert function with retry on HTTP 429."
```

---

## Lifecycle Automation (`start_kaggle.py`)

You do not need to manually check logs or copy URLs when things reset:

### 1. One-Shot Sync
Whenever you want to use the server:
```powershell
python start_kaggle.py
```
- **If active:** Tests `/health` and confirms readiness in 1 second.
- **If 4-hour Cloudflare tunnel expired:** Kaggle supervisor auto-rotates the tunnel and broadcasts the new URL to `ntfy.sh`. `start_kaggle.py` pulls the URL, tests it, and updates `.env` without restarting the session.
- **If 12-hour Kaggle session expired:** Pushes `kaggle_notebook/swift_server.py`, waits for weights to load, pulls the new URL, and syncs `.env`.

### 2. Continuous Background Guardian
```powershell
python start_kaggle.py --watch
```
Runs silently in the background, testing health every 60 seconds and auto-recovering if anything drops.

---

## Switching to Qwen 2.5 Coder 14B (3× Speedup)

To benchmark or switch to **Qwen 2.5 Coder 14B** (32+ tokens/sec, 0s deliberation delay):

1. **Dedicated Kaggle Script**:
   We have provided `kaggle_notebook/qwen25_coder_server.py`.
2. **Download Model on Kaggle (~90s)**:
   ```python
   !huggingface-cli download Qwen/Qwen2.5-Coder-14B-Instruct-GGUF qwen2.5-coder-14b-instruct-q4_k_m.gguf --local-dir /kaggle/working/qwen-model --local-dir-use-symlinks False
   ```
3. **Run Server**:
   Start `llama-server` with `-m /kaggle/working/qwen-model/qwen2.5-coder-14b-instruct-q4_k_m.gguf -c 8192`.
   The Windows client and Cloudflare tunnel work identically!

---

## Connecting to AI Editors

### Cursor
1. **Settings** → **Models** → **OpenAI API Key**.
2. **Base URL**: `https://<your-tunnel>.trycloudflare.com/v1`
3. **API Key**: `my_secret` (or value of `QWEN_API_KEY` from `.env`)
4. **Model Name**: `qwen`

### Continue.dev (VS Code / JetBrains)
In `~/.continue/config.json`:
```json
{
  "models": [
    {
      "title": "Kaggle Qwen",
      "provider": "openai",
      "model": "qwen",
      "apiBase": "https://<your-tunnel>.trycloudflare.com/v1",
      "apiKey": "my_secret"
    }
  ]
}
```

---

## Open-Source & Security Assurance

- **Zero Credentials in Git**: `.env`, `.env.bak*`, `backup_working_setup/`, and local caches are strictly ignored by `.gitignore`.
- **Sanitized Templates**: `.env.example` contains only placeholder values.
- **Automated Masking**: Tokens are automatically masked in CLI outputs (`my_s...cret`).

---

## License
MIT License. Free for personal, research, and commercial use.
