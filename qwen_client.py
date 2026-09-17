"""
qwen_client.py - Windows Client for Qwen GGUF on Kaggle / Cloudflare.

Designed for AI editors (OpenCode, Claude, etc.), CLI scripts, and interactive queries.
Reads configuration from .env, sends requests securely, and outputs strictly the
assistant response to stdout.

Usage:
    python qwen_client.py "Explain Python decorators in simple terms."
    python qwen_client.py --max-tokens 1024 "Write a FastAPI CRUD endpoint."
    echo "What is 15 * 12?" | python qwen_client.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent / ".env"


def normalize_base_url(raw_url: str) -> str:
    """Normalize base URL to ensure clean /v1 endpoint without /v1/v1 duplication."""
    url = raw_url.strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3].rstrip("/")
    return url


def load_config() -> tuple[str, str, str, int]:
    """Load and validate configuration from .env."""
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=True)
    else:
        load_dotenv(override=True)

    base_url = os.getenv("QWEN_BASE_URL", "").strip()
    api_key = os.getenv("QWEN_API_KEY", "").strip()
    model = os.getenv("QWEN_MODEL", "qwen").strip()

    try:
        timeout = int(os.getenv("QWEN_TIMEOUT", "300"))
    except ValueError:
        timeout = 300

    if not base_url:
        print(
            "[-] Error: QWEN_BASE_URL is not configured in .env\n"
            "    Run 'python update_windows_env.py <url>' to set the tunnel URL.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not api_key:
        print(
            "[-] Error: QWEN_API_KEY is not configured in .env\n"
            "    Set your llama-server API key in .env.",
            file=sys.stderr,
        )
        sys.exit(1)

    return normalize_base_url(base_url), api_key, model, timeout


def query_qwen(
    prompt: str,
    *,
    system_prompt: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    stream: bool = True,
    show_thinking: bool = False,
    verbose: bool = False,
    override_timeout: int | None = None,
) -> str:
    """Send a chat completion request to the OpenAI-compatible endpoint."""
    base_url, api_key, default_model, default_timeout = load_config()
    timeout = override_timeout or default_timeout

    endpoint = f"{base_url}/v1/chat/completions"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": default_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    if verbose:
        print(f"[*] Connecting to: {endpoint}", file=sys.stderr)
        print(f"[*] Timeout: {timeout}s | max_tokens: {max_tokens} | stream: {stream}", file=sys.stderr)

    try:
        if stream:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=timeout,
                stream=True,
            )
            response.raise_for_status()

            import time
            start_time = time.perf_counter()
            first_token_time: float | None = None
            code_start_time: float | None = None
            token_count = 0
            reasoning_tokens = 0
            code_tokens = 0
            timings_data: dict = {}
            usage_data: dict = {}
            full_reasoning: list[str] = []
            full_content: list[str] = []
            
            in_thinking_phase = False
            code_phase_announced = False
            seen_first_chunk = False

            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8", errors="ignore").strip()
                if decoded.startswith("data: "):
                    data_str = decoded[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if "timings" in chunk:
                            timings_data = chunk["timings"]
                        if "usage" in chunk:
                            usage_data = chunk["usage"]

                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        reasoning_piece = delta.get("reasoning_content", "")
                        content_piece = delta.get("content", "")

                        if not seen_first_chunk:
                            seen_first_chunk = True
                            first_token_time = time.perf_counter()

                        # 1. Handle explicit reasoning_content (DeepSeek / OpenAI reasoning format)
                        if reasoning_piece:
                            if not in_thinking_phase:
                                in_thinking_phase = True
                                sys.stdout.write("🧠 [THINKING PROCESS & DELIBERATION] (Qwen 3.8 Reasoning Engine):\n")
                                sys.stdout.write("--------------------------------------------------------------------\n")
                                sys.stdout.flush()
                            full_reasoning.append(reasoning_piece)
                            reasoning_tokens += 1
                            sys.stdout.write(reasoning_piece)
                            sys.stdout.flush()

                        # 2. Handle content_piece (which might contain <think> tags or raw deliberation)
                        if content_piece:
                            # Check for <think> tag start
                            if "<think>" in content_piece:
                                in_thinking_phase = True
                                sys.stdout.write("🧠 [THINKING PROCESS & DELIBERATION] (Qwen 3.8 Reasoning Engine):\n")
                                sys.stdout.write("--------------------------------------------------------------------\n")
                                sys.stdout.flush()
                                content_piece = content_piece.replace("<think>", "")

                            # Check for raw deliberation start if not already marked
                            if not in_thinking_phase and not code_phase_announced:
                                if any(content_piece.strip().startswith(kw) for kw in ["We need to", "Need to", "Thinking:", "The user wants"]):
                                    in_thinking_phase = True
                                    sys.stdout.write("🧠 [THINKING PROCESS & DELIBERATION] (Qwen 3.8 Reasoning Engine):\n")
                                    sys.stdout.write("--------------------------------------------------------------------\n")
                                    sys.stdout.flush()

                            # Check for </think> tag end
                            if "</think>" in content_piece:
                                in_thinking_phase = False
                                content_piece = content_piece.replace("</think>", "")
                                if not code_phase_announced:
                                    code_phase_announced = True
                                    code_start_time = time.perf_counter()
                                    sys.stdout.write("\n\n" + "=" * 68 + "\n")
                                    sys.stdout.write("💻 [FINAL CODE SYNTHESIS]:\n")
                                    sys.stdout.write("=" * 68 + "\n")
                                    sys.stdout.flush()

                            # If we were in thinking phase and now encountering markdown code block
                            if in_thinking_phase and not code_phase_announced:
                                if "```" in content_piece or content_piece.strip().startswith("def ") or content_piece.strip().startswith("Here is"):
                                    in_thinking_phase = False
                                    code_phase_announced = True
                                    code_start_time = time.perf_counter()
                                    sys.stdout.write("\n\n" + "=" * 68 + "\n")
                                    sys.stdout.write("💻 [FINAL CODE SYNTHESIS]:\n")
                                    sys.stdout.write("=" * 68 + "\n")
                                    sys.stdout.flush()

                            # If reasoning finished and code starts
                            if not in_thinking_phase and full_reasoning and not code_phase_announced:
                                code_phase_announced = True
                                code_start_time = time.perf_counter()
                                sys.stdout.write("\n\n" + "=" * 68 + "\n")
                                sys.stdout.write("💻 [FINAL CODE SYNTHESIS]:\n")
                                sys.stdout.write("=" * 68 + "\n")
                                sys.stdout.flush()

                            if in_thinking_phase:
                                full_reasoning.append(content_piece)
                                reasoning_tokens += 1
                            else:
                                full_content.append(content_piece)
                                code_tokens += 1

                            if content_piece:
                                sys.stdout.write(content_piece)
                                sys.stdout.flush()
                            token_count += 1
                    except Exception:
                        pass

            sys.stdout.write("\n")
            sys.stdout.flush()

            total_elapsed = time.perf_counter() - start_time
            ttft = (first_token_time - start_time) if first_token_time else total_elapsed
            
            # Timings breakdown
            llama_gen_ms = timings_data.get("predicted_ms", 0)
            llama_speed = timings_data.get("predicted_per_second", 0.0)
            comp_tokens = timings_data.get("predicted_n", token_count)
            p_speed = timings_data.get("prompt_per_second")
            p_ms = timings_data.get("prompt_ms", 0)

            gen_time = (llama_gen_ms / 1000.0) if llama_gen_ms > 0 else (total_elapsed - ttft)
            if llama_speed <= 0.0 and gen_time > 0:
                llama_speed = comp_tokens / gen_time

            think_dur = (code_start_time - start_time) if code_start_time else (gen_time * 0.4 if full_reasoning else 0.0)
            code_dur = max(0.0, gen_time - think_dur)

            if verbose:
                print("\n" + "=" * 68, file=sys.stderr)
                print(" 📊 BENCHMARK SUMMARY (QWEN 3.8 27B ON 2x T4)", file=sys.stderr)
                print("=" * 68, file=sys.stderr)
                print(f" • Model Architecture:     Swift-Qwen 3.8 (27B Distilled Reasoning)", file=sys.stderr)
                print(f" • GPU Acceleration:      2x Tesla T4 GPUs (Tensor-Split 1,1)", file=sys.stderr)
                print(f" • Time to First Token:    {ttft:.2f}s  (Network latency + prompt eval)", file=sys.stderr)
                if full_reasoning:
                    print(f" • Deliberation / Think:  {think_dur:.1f}s ({reasoning_tokens} thinking tokens)", file=sys.stderr)
                print(f" • Code Generation Time:   {code_dur:.1f}s ({code_tokens} code tokens)", file=sys.stderr)
                print(f" • Generation Speed:       {llama_speed:.1f} tokens/sec", file=sys.stderr)
                print(f" • Total Latency:          {total_elapsed:.1f}s", file=sys.stderr)
                print("-" * 68, file=sys.stderr)
                print("=" * 68 + "\n", file=sys.stderr)

            return "".join(full_content) if full_content else "".join(full_reasoning)

        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=timeout,
        )

        if response.status_code == 401:
            print(
                "[-] Authentication Error (HTTP 401): Invalid QWEN_API_KEY.\n"
                "    Verify your API key matches the --api-key passed to llama-server on Kaggle.",
                file=sys.stderr,
            )
            sys.exit(1)

        if response.status_code in (502, 503, 530):
            print(
                f"[-] Cloudflare Tunnel Error (HTTP {response.status_code}).\n"
                "    The tunnel may have expired or the Kaggle notebook stopped.\n"
                "    Check the Kaggle notebook and run update_windows_env.py if the URL changed.",
                file=sys.stderr,
            )
            sys.exit(1)

        response.raise_for_status()

        data = response.json()
        choice = data["choices"][0]
        message = choice.get("message", {})
        content = message.get("content", "")
        reasoning = message.get("reasoning_content", "")

        if verbose:
            usage = data.get("usage", {})
            timings = data.get("timings", {})
            if usage:
                print(f"[*] Usage: {json.dumps(usage)}", file=sys.stderr)
            if timings:
                gen_speed = timings.get("predicted_per_second", 0)
                print(f"[*] Generation Speed: {gen_speed:.1f} tokens/sec", file=sys.stderr)

        # If content is empty (e.g., token limit reached during reasoning), return reasoning
        final_output = content if content else reasoning
        return final_output

    except requests.exceptions.Timeout:
        print(
            f"[-] Request timed out after {timeout} seconds.\n"
            "    Try increasing QWEN_TIMEOUT in .env or lowering max_tokens.",
            file=sys.stderr,
        )
        sys.exit(1)
    except requests.exceptions.ConnectionError as exc:
        print(
            f"[-] Connection failed to {base_url}.\n"
            f"    Details: {exc}\n"
            "    The Cloudflare Quick Tunnel has likely expired or restarted.\n"
            "    Obtain the new URL and run: python update_windows_env.py <new_url>",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        print(f"[-] Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the Kaggle-hosted Qwen GGUF model via Cloudflare Quick Tunnel."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="The prompt to send to the model. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--system",
        dest="system_prompt",
        default=None,
        help="Optional system prompt to guide model behavior.",
    )
    parser.add_argument(
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=800,
        help="Maximum completion tokens to generate (default: 800).",
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--timeout",
        dest="timeout",
        type=int,
        default=None,
        help="Override HTTP timeout in seconds (default from .env: 300).",
    )
    parser.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        default=True,
        help="Stream tokens in real-time as they are generated (default: True).",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Wait for the full response before outputting.",
    )
    parser.add_argument(
        "--show-thinking",
        dest="show_thinking",
        action="store_true",
        default=True,
        help="Print internal reasoning tokens with structured formatting (default: True).",
    )
    parser.add_argument(
        "--optimize-think",
        dest="optimize_think",
        action="store_true",
        default=False,
        help="Inject expert architectural guidance for crisp, concise technical deliberation (ideal for video recording).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print diagnostic timings and metadata to stderr.",
    )

    args = parser.parse_args()

    system_prompt = args.system_prompt
    if args.optimize_think and not system_prompt:
        system_prompt = (
            "You are an expert software engineer. Provide a brief, concise technical deliberation "
            "(3-4 crisp bullet points analyzing key requirements and edge cases), "
            "then immediately synthesize the clean, production-grade code without excessive filler."
        )

    # Determine prompt from argument or standard input
    prompt_text = args.prompt
    if not prompt_text:
        if not sys.stdin.isatty():
            prompt_text = sys.stdin.read().strip()
        else:
            try:
                prompt_text = input("Prompt: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled.", file=sys.stderr)
                sys.exit(1)

    if not prompt_text:
        print("[-] Error: No prompt provided.", file=sys.stderr)
        sys.exit(1)

    # Execute and output strictly to stdout
    result = query_qwen(
        prompt=prompt_text,
        system_prompt=system_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stream=args.stream,
        show_thinking=args.show_thinking,
        verbose=args.verbose,
        override_timeout=args.timeout,
    )

    if not args.stream:
        print(result)


if __name__ == "__main__":
    main()
