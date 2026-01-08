#!/usr/bin/env python3
"""Loreguard CLI - Standalone headless mode for embedding in games.

Usage:
    loreguard-cli --token lg_xxx... --model /path/to/model.gguf
    loreguard-cli --token lg_xxx... --model-id qwen3-4b
    loreguard-cli --token lg_xxx... --worker-id my-pc --model-id qwen3-4b

Environment variables (alternative to args):
    LOREGUARD_TOKEN     API token
    LOREGUARD_MODEL     Path to model file
    LOREGUARD_MODEL_ID  Model ID to download (if not using custom model)
    LOREGUARD_PORT      Local llama-server port (default: 8080)
    LOREGUARD_BACKEND   Backend URL (default: wss://api.loreguard.com/workers)
    LOREGUARD_WORKER_ID Worker ID (default: hostname)
"""

import argparse
import asyncio
import logging
import os
import signal
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("loreguard")


class LoreguardCLI:
    """Headless Loreguard client for embedding in games."""

    def __init__(
        self,
        token: str,
        model_path: Optional[Path] = None,
        model_id: Optional[str] = None,
        port: int = 8080,
        backend_url: str = "wss://api.loreguard.com/workers",
        worker_id: Optional[str] = None,
    ):
        self.token = token
        self.model_path = model_path
        self.model_id = model_id
        self.port = port
        self.backend_url = backend_url
        # Worker ID: use provided value, or default to hostname
        self.worker_id = worker_id or socket.gethostname() or "worker"

        self._llama = None
        self._tunnel = None
        self._sdk_port: Optional[int] = None
        self._running = False

        # Metrics
        self._requests = 0
        self._tokens = 0
        self._start_time = None

    async def run(self) -> int:
        """Run the client. Returns exit code."""
        self._start_time = datetime.now()

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self._shutdown()))

        log.info("=" * 50)
        log.info("Loreguard CLI - Starting")
        log.info("=" * 50)

        try:
            # Resolve model path
            if not await self._resolve_model():
                return 1

            # Start llama-server
            if not await self._start_llama_server():
                return 1

            # Connect to backend
            if not await self._connect_backend():
                return 1

            self._running = True
            log.info("=" * 50)
            log.info("Ready! Waiting for inference requests...")
            log.info("Press Ctrl+C to stop")
            log.info("=" * 50)

            # Keep running until shutdown
            while self._running:
                await asyncio.sleep(1)
                self._log_stats()

            return 0

        except Exception as e:
            log.error(f"Fatal error: {e}")
            return 1
        finally:
            await self._cleanup()

    async def _resolve_model(self) -> bool:
        """Resolve model path, downloading if needed."""
        if self.model_path:
            if not self.model_path.exists():
                log.error(f"Model not found: {self.model_path}")
                return False
            log.info(f"Using model: {self.model_path}")
            return True

        if self.model_id:
            from .models_registry import SUPPORTED_MODELS
            from .llama_server import get_models_dir, DownloadProgress

            # Find model by ID
            model = None
            for m in SUPPORTED_MODELS:
                if m.id == self.model_id:
                    model = m
                    break

            if not model:
                log.error(f"Unknown model ID: {self.model_id}")
                log.info("Available models:")
                for m in SUPPORTED_MODELS:
                    log.info(f"  - {m.id}: {m.name}")
                return False

            models_dir = get_models_dir()
            self.model_path = models_dir / model.filename

            if self.model_path.exists():
                log.info(f"Model already downloaded: {self.model_path}")
                return True

            # Download
            log.info(f"Downloading {model.name} ({model.size_gb:.1f} GB)...")
            try:
                await self._download_model(model, self.model_path)
                log.info(f"Download complete: {self.model_path}")
                return True
            except Exception as e:
                log.error(f"Download failed: {e}")
                return False

        log.error("No model specified. Use --model or --model-id")
        return False

    async def _download_model(self, model, dest: Path) -> None:
        """Download a model file with progress."""
        import httpx

        dest.parent.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
            async with client.stream("GET", model.url) as response:
                response.raise_for_status()
                total = model.size_bytes or int(response.headers.get("content-length", 0))
                downloaded = 0
                last_log = 0

                with open(dest, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Log progress every 10%
                        pct = int(downloaded / total * 100) if total else 0
                        if pct >= last_log + 10:
                            last_log = pct
                            log.info(f"  {pct}% ({downloaded // 1024 // 1024} MB)")

    async def _start_llama_server(self) -> bool:
        """Start llama-server."""
        from .llama_server import (
            LlamaServerProcess,
            is_llama_server_installed,
            download_llama_server,
            DownloadProgress,
        )

        # Download llama-server if needed
        if not is_llama_server_installed():
            log.info("Downloading llama-server...")
            try:
                def on_progress(msg: str, progress: DownloadProgress | None):
                    if progress and int(progress.percent) % 20 == 0:
                        log.info(f"  {int(progress.percent)}%")
                await download_llama_server(on_progress)
                log.info("llama-server downloaded")
            except Exception as e:
                log.error(f"Failed to download llama-server: {e}")
                return False

        # Start server
        log.info(f"Starting llama-server on port {self.port}...")
        try:
            self._llama = LlamaServerProcess(self.model_path, port=self.port)
            self._llama.start()

            # Wait for ready
            ready = await self._llama.wait_for_ready(timeout=120.0)
            if not ready:
                log.error("llama-server failed to start (timeout)")
                return False

            log.info("llama-server ready")
            return True

        except Exception as e:
            log.error(f"Failed to start llama-server: {e}")
            return False

    async def _connect_backend(self) -> bool:
        """Connect to Loreguard backend."""
        # Dev mode - skip backend connection
        if self.token == "dev_mock_token":
            log.info("DEV MODE: Skipping backend connection")
            log.info(f"llama-server running at http://127.0.0.1:{self.port}")
            log.info("You can send requests directly to the llama-server")
            return True

        from .tunnel import BackendTunnel
        from .llm import LLMProxy

        log.info(f"Connecting to {self.backend_url}...")
        log.info(f"Worker ID: {self.worker_id}")

        try:
            llm_proxy = LLMProxy(f"http://127.0.0.1:{self.port}")

            self._tunnel = BackendTunnel(
                backend_url=self.backend_url,
                llm_proxy=llm_proxy,
                worker_id=self.worker_id,
                worker_token=self.token,
                model_id=self.model_path.stem if self.model_path else "unknown",
            )

            self._tunnel.on_request_complete = self._on_request_complete

            # Start SDK server for local game clients
            from .http_server import start_sdk_server
            try:
                self._sdk_port = start_sdk_server(
                    tunnel=self._tunnel,
                    main_loop=asyncio.get_running_loop(),
                )
                log.info(f"SDK server listening on 127.0.0.1:{self._sdk_port}")
            except Exception as e:
                log.error(f"Failed to start SDK server: {e}")
                return False

            # Start connection (runs in background)
            asyncio.create_task(self._tunnel.connect())

            # Wait a bit for connection
            await asyncio.sleep(2)
            log.info("Backend connection established")
            return True

        except Exception as e:
            log.error(f"Failed to connect to backend: {e}")
            return False

    def _on_request_complete(
        self, npc: str, tokens: int, ttft_ms: float, total_ms: float
    ) -> None:
        """Called when a request completes."""
        self._requests += 1
        self._tokens += tokens
        tps = (tokens / total_ms * 1000) if total_ms > 0 else 0

        log.info(
            f"Request #{self._requests}: {npc} | "
            f"{tokens} tokens | {ttft_ms:.0f}ms TTFT | "
            f"{total_ms/1000:.1f}s total | {tps:.1f} tk/s"
        )

    def _log_stats(self) -> None:
        """Log periodic stats (every 60 seconds)."""
        if not self._start_time:
            return

        elapsed = (datetime.now() - self._start_time).total_seconds()
        if int(elapsed) % 60 == 0 and int(elapsed) > 0:
            mins = int(elapsed // 60)
            log.info(
                f"Stats: {mins}m uptime | "
                f"{self._requests} requests | "
                f"{self._tokens:,} tokens"
            )

    async def _shutdown(self) -> None:
        """Graceful shutdown."""
        if not self._running:
            return

        log.info("Shutting down...")
        self._running = False

    async def _cleanup(self) -> None:
        """Cleanup resources."""
        if self._tunnel:
            try:
                await self._tunnel.disconnect()
            except Exception:
                pass
        # Stop SDK server (cleans up runtime.json)
        try:
            from .http_server import stop_sdk_server
            stop_sdk_server()
        except Exception:
            pass

        if self._llama:
            try:
                self._llama.stop()
            except Exception:
                pass

        log.info("Goodbye!")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Loreguard CLI - Local inference for game NPCs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  loreguard --token lg_xxx... --model ./model.gguf
  loreguard --token lg_xxx... --model-id qwen3-4b
  loreguard --token lg_xxx... --worker-id my-pc --model-id qwen3-4b
  loreguard --chat --token lg_xxx...   # Test NPC chat (no model needed)
  loreguard --dev --model-id qwen3-4b  # Local dev mode

Available model IDs:
  qwen3-4b-instruct    Qwen3 4B Instruct (recommended, 2.8 GB)
  llama-3.2-3b         Llama 3.2 3B Instruct (2.0 GB)
  qwen3-8b             Qwen3 8B (5.2 GB)
  meta-llama-3-8b      Meta Llama 3 8B (4.9 GB)
        """,
    )

    parser.add_argument(
        "--token",
        default=os.getenv("LOREGUARD_TOKEN", ""),
        help="API token (or set LOREGUARD_TOKEN env var)",
    )
    parser.add_argument(
        "--worker-id",
        default=os.getenv("LOREGUARD_WORKER_ID", ""),
        help="Worker ID (default: hostname)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=os.getenv("LOREGUARD_MODEL"),
        help="Path to .gguf model file",
    )
    parser.add_argument(
        "--model-id",
        default=os.getenv("LOREGUARD_MODEL_ID"),
        help="Model ID to download (e.g., qwen3-4b-instruct)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("LOREGUARD_PORT", "8080")),
        help="Local llama-server port (default: 8080)",
    )
    parser.add_argument(
        "--backend",
        default=os.getenv("LOREGUARD_BACKEND", "wss://api.loreguard.com/workers"),
        help="Backend WebSocket URL",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging and show pipeline pass updates (in wizard mode)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Dev mode - skip backend connection, just run llama-server",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Chat mode - test NPC chat via Loreguard API (no model needed)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Chat mode - test NPC chat directly via API (no model needed)
    if args.chat:
        if not args.token:
            log.error("Token required for chat mode. Use --token or set LOREGUARD_TOKEN")
            sys.exit(1)

        log.info("=" * 50)
        log.info("Loreguard Chat Mode - Testing NPC Pipeline")
        log.info("=" * 50)

        from .npc_chat import run_npc_chat
        try:
            # Note: verbose mode without tunnel will request pass updates
            # but won't display them (no WebSocket connection in chat mode)
            asyncio.run(run_npc_chat(api_token=args.token, verbose=args.verbose))
        except KeyboardInterrupt:
            pass
        sys.exit(0)

    # Dev mode - skip token validation
    if args.dev:
        args.token = "dev_mock_token"
        log.info("Running in DEV MODE - no backend connection")
    else:
        # Validate token is present (server will validate format)
        if not args.token:
            log.error("Token required. Use --token or set LOREGUARD_TOKEN (or use --dev)")
            sys.exit(1)

    # Validate model
    if not args.model and not args.model_id:
        log.error("Model required. Use --model or --model-id")
        sys.exit(1)

    # Run
    cli = LoreguardCLI(
        token=args.token,
        model_path=args.model,
        model_id=args.model_id,
        port=args.port,
        backend_url=args.backend,
        worker_id=args.worker_id or None,  # None will use hostname
    )

    exit_code = asyncio.run(cli.run())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
