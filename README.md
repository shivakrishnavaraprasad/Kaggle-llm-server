# ⚡ Kaggle Swift API (Part 1): Free Self-Hosted 27B Reasoning LLM on Dual Cloud GPUs

Deploy and run a frontier **27B Reasoning Model (Swift-Qwen 3.8 27B, 16.8 GB)** on **2 × NVIDIA Tesla T4 GPUs (32 GB VRAM Total)** on Kaggle for **100% free**.

Exposes an authenticated, **OpenAI-compatible `/v1` endpoint** through an automated **Cloudflare Quick Tunnel**, connecting directly to your Windows terminal, Python scripts, and AI coding editors (**Cursor, Claude Code, Continue.dev, Aider**).

---

## 💡 The Problem & The Journey (From LinkedIn Part 1)

Running large open-source language models locally is impossible on a standard machine with 8 GB RAM. Kaggle provides **2 × Tesla T4 GPUs (32 GB VRAM)** for free, but doing this manually came with frustrating friction:

- **Hardware & VRAM Constraints**: A 27B model requires multi-GPU tensor splitting (`1,1`) across both T4 GPUs.
- **Native Library & ABI Mismatches**: Precompiled binaries crash without matching dynamic libraries (`libmtmd.so`) built for Kaggle's Linux runtime.
- **Networking & Tunnel Drops**: Cloudflare Quick Tunnels reset every 4 hours, and Kaggle kernels expire after 12 hours.
- **Manual Overhead**: Searching through hundreds of lines of startup logs, guessing when the model is ready, and manually copying tunnel URLs back and forth.

Instead of solving each problem manually every time, this repository automates the entire lifecycle:

```text
Start Model → Verify Health → Establish Cloudflare Tunnel → Broadcast URL → Sync Local .env → Test APIs
```

---

## 📊 Live Verified Benchmarks on 2 × Tesla T4 (32 GB VRAM)

All performance metrics below are measured live via `qwen_client.py` and `test_health.py` over an active Cloudflare Quick Tunnel:

| Metric | Measured Value | Notes |
|---|---|---|
| **Model** | **Swift-Qwen 3.8 27B** (`Q4_K_M`) | Distilled reasoning architecture |
| **Model Size** | **16.80 GB** | Loaded across 2× Tesla T4 GPUs (VRAM: 32 GB) |
| **Context Window** | **4,096 tokens** | Full multi-turn reasoning context |
| **Generation Speed** | **~12.1 tokens/sec** | Measured during active streaming output |
| **Deliberation / Thinking Delay** | **12–16 seconds** | Dedicated reasoning phase (`<think>` tokens) |
| **Total Response Turnaround** | **20–40 seconds** | Complete end-to-end response delivery |
| **Time-To-First-Token (TTFT)** | **~1.4s** | Initial TCP + tunnel handshake |
| **Cost** | **$0.00 (100% Free)** | Kaggle 30h/week GPU quota |

---

## 📋 Prerequisites

1. **Free Kaggle Account**:
   - Register at [kaggle.com](https://www.kaggle.com) and verify your phone number (required to enable **GPU T4 x2** and **Internet**).
2. **Kaggle API Token**:
   - Go to `https://www.kaggle.com/settings` → **API** → **Create New Token**.
   - Note your `username` and API `key` / `token`.
3. **Local Machine**:
   - Windows 10/11 (or Linux/macOS).
   - Python 3.10+ installed.

---

## 🚀 Quick Start Guide

### Step 1: Clone Repository & Set Up Environment

Open PowerShell on Windows:

```powershell
# 1. Clone repository
git clone https://github.com/shivakrishnavaraprasad/Kaggle-llm-server.git
cd Kaggle-llm-server

# 2. Create and activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your local .env
cp .env.example .env
```

### Step 2: Configure Local Credentials (`.env`)

Open `.env` in any editor and fill in your Kaggle details:

```ini
# ==============================================================================
# 1. KAGGLE CREDENTIALS
# ==============================================================================
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_API_TOKEN=your_kaggle_token_here
KAGGLE_KERNEL_SLUG=your_kaggle_username/swift-server

# ==============================================================================
# 2. SERVER & SECURITY CONFIGURATION
# ==============================================================================
QWEN_API_KEY=my_secret
OPENAI_MODEL=qwen
QWEN_TIMEOUT=300
QWEN_SYNC_TOPIC=kaggle_qwen_your_kaggle_username

# ==============================================================================
# 3. TUNNEL URL (Auto-updated by start_kaggle.py - leave blank initially)
# ==============================================================================
QWEN_BASE_URL=
```

### Step 3: Deploy & Launch with One Command

Run the single-script lifecycle manager:

```powershell
.\.venv\Scripts\python.exe start_kaggle.py --restart
```

#### What Happens Automatically:
1. Pushes `kaggle_notebook/swift_server.py` and attachments to Kaggle.
2. Boots dual Tesla T4 GPUs and compiles native `libmtmd.so` to prevent Linux ABI crashes.
3. Loads the 16.8 GB model weights into 2x T4 VRAM with multi-GPU tensor splitting (`1,1`).
4. Spawns `llama-server` and establishes a Cloudflare Quick Tunnel.
5. Broadcasts the public URL via `ntfy.sh`, catches it locally, auto-updates `QWEN_BASE_URL` in `.env`, and verifies `/health`.

---

## 🧪 Testing Health & Querying the Model

### 1. Fast Health & Latency Probe
```powershell
.\.venv\Scripts\python.exe test_health.py
```
*Expected output:*
```text
[+] Health Status: HTTP 200 OK (0.88s) | Response: {"status":"ok"}
[+] Chat Response: HTTP 200 OK (4.44s)
    Assistant: Health check confirmed.
    Speed:     12.1 tokens/sec
```

### 2. Dual-Phase Streaming Query (with Thinking Process)
```powershell
.\.venv\Scripts\python.exe qwen_client.py "Write a Python function to reverse a linked list."
```

*Output shows the real-time deliberation trace and the final code synthesis:*
```text
🧠 [THINKING PROCESS & DELIBERATION] (Qwen 3.8 Reasoning Engine):
Analyzing linked list reversal... Need prev, current, next pointers...

💻 [FINAL CODE SYNTHESIS]:
def reverse_linked_list(head):
    prev = None
    curr = head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev
```

---

## 🔄 Autonomous Lifecycle Management (`start_kaggle.py`)

You never need to babysit the Kaggle browser tab or manually copy URLs:

| Command | Action |
|---|---|
| `python start_kaggle.py` | **One-Shot Smart Sync**: Probes current tunnel. If active, exits in 1s. If the 4-hour Cloudflare tunnel expired, extracts the rotated URL from `ntfy.sh` and syncs `.env` without rebooting Kaggle. |
| `python start_kaggle.py --restart` | **Force Restart**: Re-pushes code, launches fresh Kaggle GPU session, polls for readiness, and updates `.env`. |
| `python start_kaggle.py --status` | **Status Check**: Queries remote kernel execution status and recent logs via Kaggle API. |
| `python start_kaggle.py --watch` | **Background Guardian**: Pings `/health` every 60s. Auto-detects 4h tunnel drops and 12h session resets, restoring connectivity automatically. |

---

## 🔌 Connecting to Local AI Editors

Your Kaggle server exposes a standard OpenAI-compatible `/v1` endpoint:

### Cursor
1. Go to **Settings** → **Models** → **OpenAI API Key**.
2. **Base URL**: `https://<your-tunnel>.trycloudflare.com/v1`
3. **API Key**: `my_secret` (or value of `QWEN_API_KEY` from `.env`)
4. **Model Name**: `qwen`

### Continue.dev (VS Code / JetBrains)
In `~/.continue/config.json`:
```json
{
  "models": [
    {
      "title": "Kaggle Swift Qwen 3.8",
      "provider": "openai",
      "model": "qwen",
      "apiBase": "https://<your-tunnel>.trycloudflare.com/v1",
      "apiKey": "my_secret"
    }
  ]
}
```

---

## 📁 Repository Structure (Part 1)

```text
Kaggle-llm-server/
│
├── .env.example                       # Local environment variables template
├── .gitignore                         # Protects secrets, credentials, and local backups
├── requirements.txt                   # Python dependencies (requests, kaggle, etc.)
├── README.md                          # Part 1 documentation and architecture
├── PROJECT_AUDIT.md                   # Engineering audit and security validation
│
├── start_kaggle.py                    # Single-script lifecycle manager & auto-recovery guardian
├── update_windows_env.py              # Atomic updater for QWEN_BASE_URL in .env with backups
├── test_health.py                     # Health probe and latency benchmark utility
├── test_qwen_api.py                   # OpenAI API compatibility test suite
├── qwen_client.py                     # CLI streaming client with dual-phase reasoning display
├── openai_compatible_client.py        # OpenAI Python SDK integration example
│
└── kaggle_notebook/
    ├── kernel-metadata.json           # Kaggle GPU kernel definition and dataset attachments
    ├── swift_server.py                # Kaggle bootstrap supervisor for Qwen 3.8 27B
    └── stop_server.py                 # Graceful termination utility
```

---

## 🔒 Security Assurance

- **Zero Hardcoded Secrets**: All API tokens and passwords reside exclusively in your local `.env`.
- **Automatic Token Masking**: Sensitive keys are masked in CLI logs (`my_s...cret`).
- **No Binaries in Git**: Heavy `.gguf` weights (16.8 GB) and CUDA binaries are mounted via Kaggle datasets.

---

## 🔭 What's Next (Preview of Part 2)

While the 27B reasoning model provides deep logic and mathematical reasoning, the **12–16 second deliberation delay** makes it slow for interactive tab-completions in coding editors.

In **Part 2**, we test:
- **Qwen 2.5 Coder 32B**: Instant start with SWE-Bench level coding.
- **Qwen 2.5 Coder 14B**: 2.2x speedup (24+ tokens/sec, 0s delay) for high-FPS coding.
- **Speculative Decoding**: Pairing a 14B model with a 0.5B draft model, and why Tesla T4 memory bandwidth limits speculative verification.

Stay tuned!

---

## License
MIT License. Free for personal, research use.
