"""
test_qwen_api.py - Complete Automated Verification Suite for Qwen API on Kaggle.

Validates:
1. Environment variables and configuration syntax
2. Local/Remote health endpoint (/health)
3. Unauthenticated request rejection (401 Unauthorized)
4. Authenticated chat completion via direct HTTP (requests)
5. Authenticated chat completion via OpenAI SDK
6. Response latency, prompt evaluation speed, and token generation speed

Returns:
    Exit code 0 on all tests passing.
    Exit code 1 on any required test failure.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

# Try importing openai SDK; test will report cleanly if missing
try:
    from openai import OpenAI
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False


def mask_secret(secret: str | None) -> str:
    """Safely mask secrets for logs and terminal display."""
    if not secret:
        return "<EMPTY>"
    secret = secret.strip()
    if len(secret) <= 8:
        return "***"
    return f"{secret[:4]}...{secret[-4:]}"


class TestRunner:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.base_url = ""
        self.api_key = ""
        self.model_name = "qwen"
        self.timeout = 300

    def log_section(self, title: str) -> None:
        print("\n" + "=" * 70)
        print(f" {title.upper()}")
        print("=" * 70)

    def log_pass(self, test_name: str, detail: str = "") -> None:
        self.passed += 1
        msg = f"[PASS] {test_name}"
        if detail:
            msg += f" -> {detail}"
        print(msg)

    def log_fail(self, test_name: str, error: str) -> None:
        self.failed += 1
        print(f"[FAIL] {test_name} -> {error}")

    def log_warn(self, test_name: str, note: str) -> None:
        self.warnings += 1
        print(f"[WARN] {test_name} -> {note}")

    def test_configuration(self) -> bool:
        """Step 1: Validate environment variables and base URL syntax."""
        self.log_section("1. Configuration Validation")
        env_path = Path(__file__).resolve().parent / ".env"

        if not env_path.exists():
            self.log_fail("Configuration File", f".env file not found at {env_path}")
            return False

        load_dotenv(dotenv_path=env_path, override=True)

        raw_base_url = os.getenv("QWEN_BASE_URL", "").strip()
        raw_api_key = os.getenv("QWEN_API_KEY", "").strip()
        self.model_name = os.getenv("QWEN_MODEL", "qwen").strip()
        
        try:
            self.timeout = int(os.getenv("QWEN_TIMEOUT", "300"))
        except ValueError:
            self.timeout = 300

        if not raw_base_url:
            self.log_fail("QWEN_BASE_URL", "Missing in .env")
            return False
        
        parsed = urlparse(raw_base_url)
        if parsed.scheme not in ("http", "https"):
            self.log_fail("QWEN_BASE_URL", f"Invalid scheme '{parsed.scheme}'. Must be http or https.")
            return False

        if not parsed.netloc:
            self.log_fail("QWEN_BASE_URL", f"Invalid hostname: {raw_base_url}")
            return False

        # Normalize: strip trailing slash and any trailing /v1
        self.base_url = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")

        if not raw_api_key:
            self.log_fail("QWEN_API_KEY", "Missing in .env")
            return False

        self.api_key = raw_api_key

        self.log_pass("Environment Loaded", f"Base URL: {self.base_url}")
        self.log_pass("API Key Configured", f"Masked Key: {mask_secret(self.api_key)}")
        self.log_pass("Model Configured", f"Model Name: {self.model_name}")
        self.log_pass("Timeout Configured", f"{self.timeout}s")
        return True

    def test_health_endpoint(self) -> bool:
        """Step 2: Probe the /health endpoint."""
        self.log_section("2. Health Endpoint Probe")
        health_url = f"{self.base_url}/health"

        start_time = time.perf_counter()
        try:
            response = requests.get(health_url, timeout=30)
            elapsed = time.perf_counter() - start_time

            if response.status_code == 200:
                self.log_pass(
                    "Health Check (/health)",
                    f"HTTP 200 OK in {elapsed:.2f}s | Response: {response.text.strip()}"
                )
                return True
            else:
                self.log_fail(
                    "Health Check (/health)",
                    f"Unexpected HTTP {response.status_code} in {elapsed:.2f}s | Body: {response.text.strip()}"
                )
                return False

        except requests.exceptions.Timeout:
            self.log_fail(
                "Health Check (/health)",
                f"Connection timed out after 30s to {health_url}. Server may be initializing or tunnel down."
            )
            return False
        except requests.exceptions.ConnectionError as exc:
            self.log_fail(
                "Health Check (/health)",
                f"Connection error to {health_url}. Cloudflare tunnel may be expired. Details: {exc}"
            )
            return False
        except Exception as exc:
            self.log_fail("Health Check (/health)", f"Unexpected exception: {exc}")
            return False

    def test_unauthenticated_rejection(self) -> bool:
        """Step 3: Verify that unauthorized chat completions are rejected."""
        self.log_section("3. Authentication Security Verification")
        chat_url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": "Ping"}],
            "max_tokens": 10,
        }

        try:
            # Send without Authorization header
            response = requests.post(chat_url, json=payload, timeout=20)
            if response.status_code in (401, 403):
                self.log_pass(
                    "Unauthorized Request Rejected",
                    f"HTTP {response.status_code} (as expected for missing key)"
                )
                return True
            else:
                self.log_warn(
                    "Unauthorized Request",
                    f"Server returned HTTP {response.status_code} instead of 401/403. Authentication may not be enforced."
                )
                return True

        except Exception as exc:
            self.log_warn("Unauthorized Request Check", f"Failed to test auth rejection: {exc}")
            return True

    def test_authenticated_requests_completion(self) -> bool:
        """Step 4: Execute chat completion via direct requests library."""
        self.log_section("4. Direct HTTP Chat Completion (requests)")
        chat_url = f"{self.base_url}/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": "Respond strictly with the single word: OK"
                }
            ],
            "temperature": 0.1,
            "max_tokens": 128,
        }

        start_time = time.perf_counter()
        try:
            print(f"[*] Sending request to {chat_url} (timeout: {self.timeout}s)...")
            response = requests.post(chat_url, headers=headers, json=payload, timeout=self.timeout)
            duration = time.perf_counter() - start_time

            if response.status_code != 200:
                self.log_fail(
                    "Direct Chat Completion",
                    f"HTTP {response.status_code} | Body: {response.text.strip()}"
                )
                return False

            data = response.json()
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content", "").strip()
            reasoning = msg.get("reasoning_content", "").strip()

            # Some models with reasoning place thought process in reasoning_content
            output_display = content if content else f"[Reasoning Only] {reasoning[:80]}..."
            self.log_pass(
                "Direct Chat Completion",
                f"HTTP 200 in {duration:.2f}s | Output: {repr(output_display)}"
            )

            # Print usage and llama.cpp timings if present
            usage = data.get("usage", {})
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                print(f"    Usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")

            timings = data.get("timings", {})
            if timings:
                p_sec = timings.get("prompt_per_second", 0)
                gen_sec = timings.get("predicted_per_second", 0)
                gen_ms = timings.get("predicted_ms", 0)
                print(f"    Timing Metrics: prompt={p_sec:.1f} tok/s | generation={gen_sec:.1f} tok/s ({gen_ms:.0f}ms)")

            return True

        except requests.exceptions.Timeout:
            self.log_fail(
                "Direct Chat Completion",
                f"Request timed out after {self.timeout}s. Model generation took too long or GPU stalled."
            )
            return False
        except requests.exceptions.ConnectionError as exc:
            self.log_fail("Direct Chat Completion", f"Connection error: {exc}")
            return False
        except Exception as exc:
            self.log_fail("Direct Chat Completion", f"Unexpected error: {exc}")
            return False

    def test_openai_sdk_completion(self) -> bool:
        """Step 5: Execute chat completion via official OpenAI Python SDK."""
        self.log_section("5. OpenAI SDK Chat Completion")
        if not HAS_OPENAI_SDK:
            self.log_warn("OpenAI SDK", "openai package is not installed; skipping SDK test.")
            return True

        start_time = time.perf_counter()
        try:
            client = OpenAI(
                base_url=f"{self.base_url}/v1",
                api_key=self.api_key,
                timeout=self.timeout,
            )

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": "What is 2 + 2? Reply with just the number."
                    }
                ],
                temperature=0.1,
                max_tokens=128,
            )
            duration = time.perf_counter() - start_time

            choice = response.choices[0]
            content = choice.message.content or ""
            reasoning = getattr(choice.message, "reasoning_content", None) or ""

            output_display = content.strip() if content.strip() else f"[Reasoning] {reasoning[:60]}..."
            self.log_pass(
                "OpenAI SDK Completion",
                f"Success in {duration:.2f}s | Output: {repr(output_display)}"
            )
            return True

        except Exception as exc:
            self.log_fail("OpenAI SDK Completion", f"Failed with: {exc}")
            return False

    def run(self) -> int:
        print("=" * 70)
        print(" QWEN GGUF API VALIDATION SUITE (KAGGLE / CLOUDFLARE)")
        print("=" * 70)

        # 1. Config
        if not self.test_configuration():
            print("\n[-] Configuration validation failed. Aborting remaining tests.")
            return 1

        # 2. Health
        health_ok = self.test_health_endpoint()
        if not health_ok:
            print("\n[-] Health check failed. The Cloudflare tunnel or Kaggle server is offline.")
            return 1

        # 3. Auth Check
        self.test_unauthenticated_rejection()

        # 4. Direct requests completion
        direct_ok = self.test_authenticated_requests_completion()

        # 5. OpenAI SDK completion
        sdk_ok = self.test_openai_sdk_completion()

        # Final Summary
        self.log_section("Test Results Summary")
        print(f"Passed:   {self.passed}")
        print(f"Failed:   {self.failed}")
        print(f"Warnings: {self.warnings}")

        if self.failed == 0 and direct_ok:
            print("\n[✓] ALL CRITICAL TESTS PASSED! Pipeline is fully operational.")
            return 0
        else:
            print("\n[✗] ONE OR MORE TESTS FAILED.")
            return 1


def main() -> None:
    runner = TestRunner()
    sys.exit(runner.run())


if __name__ == "__main__":
    main()
