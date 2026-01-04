"""WebSocket tunnel to Lorekeeper backend.

This is the core of the client bridge. It:
1. Maintains a persistent WebSocket connection to the backend
2. Registers as a worker with authentication
3. Receives LLM inference requests from the backend
4. Executes them on the local llama.cpp server
5. Returns results to the backend
"""

import asyncio
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

import websockets
from rich.console import Console

from .llm import LLMProxy

if TYPE_CHECKING:
    from .nli import NLIService

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class WorkerInfo:
    """Information about this worker."""
    id: str
    model_id: str
    max_tokens: int = 4096
    capabilities: list[str] = field(default_factory=lambda: ["chat", "completion"])
    address: str = "localhost:8080"


class BackendTunnel:
    """
    Manages WebSocket connection to Lorekeeper backend.

    Uses the worker registration protocol with revocable token authentication.
    The backend sends LLM requests through this tunnel, and we proxy them
    to the user's local LLM.
    """

    def __init__(
        self,
        backend_url: str,
        llm_proxy: LLMProxy,
        worker_id: str,
        worker_token: str,
        model_id: str = "default",
        nli_service: Optional["NLIService"] = None,
        log_callback: Callable[[str, str], None] | None = None,
    ):
        self.backend_url = backend_url
        self.llm_proxy = llm_proxy
        self.worker_id = worker_id
        self.worker_token = worker_token
        self.model_id = model_id
        self.nli_service = nli_service
        self.log_callback = log_callback

        self.ws: websockets.WebSocketClientProtocol | None = None
        self.connected = False
        self.registered = False
        self._reconnect_delay = 1  # Start with 1 second
        self._max_reconnect_delay = 60  # Max 60 seconds
        self._running = True
        self._shutdown_requested = False
        self._heartbeat_task: asyncio.Task | None = None

        # Callback for metrics (called after each request completes)
        # Signature: (npc_name: str, tokens: int, ttft_ms: float, total_ms: float) -> None
        self.on_request_complete: Callable[[str, int, float, float], None] | None = None

    def _log(self, message: str, level: str = "info"):
        """Log a message through callback or fallback to console."""
        if self.log_callback:
            self.log_callback(message, level)
        else:
            color_map = {
                "info": "cyan",
                "success": "green",
                "error": "red",
                "warn": "yellow",
            }
            color = color_map.get(level, "white")
            console.print(f"[{color}]{message}[/{color}]")

    async def connect(self):
        """Establish and maintain connection to backend with auto-reconnect."""
        if not self.worker_id or not self.worker_token:
            console.print("[red]Error: Worker ID and API token are required[/red]")
            console.print("[yellow]Get an API token from loreguard.com dashboard[/yellow]")
            return

        while self._running:
            try:
                await self._connect_once()
            except websockets.exceptions.InvalidStatus as e:
                # Server rejected connection (e.g., 401, 500)
                logger.error("WebSocket connection rejected: HTTP %s", e.response.status_code)
                console.print(f"[red]Connection rejected: HTTP {e.response.status_code}[/red]")
                if e.response.status_code == 401:
                    console.print("[yellow]  → Invalid or expired token[/yellow]")
                elif e.response.status_code >= 500:
                    console.print("[yellow]  → Server error (backend may be down)[/yellow]")
            except Exception as e:
                logger.error("Connection error: %s", e)
                console.print(f"[red]Connection error: {e}[/red]")

            if not self._running:
                break

            # Exponential backoff for reconnection
            console.print(f"[yellow]Reconnecting in {self._reconnect_delay}s...[/yellow]")
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)

    async def _connect_once(self):
        """Single connection attempt."""
        console.print(f"[yellow]Connecting to {self.backend_url}...[/yellow]")

        self.ws = await websockets.connect(
            self.backend_url,
            additional_headers={"Authorization": f"Bearer {self.worker_token}"},
            ping_interval=30,
            ping_timeout=10,
        )
        self.connected = True
        self._reconnect_delay = 1  # Reset on successful connection
        console.print("[green]Connected to backend![/green]")

        # Register as worker
        success = await self._register_worker()
        if not success:
            console.print("[red]Worker registration failed[/red]")
            await self.ws.close()
            self.connected = False
            return

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Start message handler
        try:
            await self._handle_messages()
        finally:
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass

    async def _register_worker(self) -> bool:
        """Register as a worker with authentication."""
        if not self.ws:
            return False

        # Use the configured model_id (already mapped to backend-accepted ID)
        # Don't override with LLM server's model name (it returns filenames like Qwen3-4B-Instruct.gguf)
        model_id = self.model_id

        # Build capabilities list - add "nli" if NLI service is available
        capabilities = ["chat", "completion"]
        if self.nli_service is not None and self.nli_service.is_loaded:
            capabilities.append("nli")
            console.print("[cyan]NLI capability enabled[/cyan]")

        # Build registration message
        register_msg = {
            "id": self._generate_message_id(),
            "type": "worker_register",
            "timestamp": self._iso_timestamp(),
            "senderId": self.worker_id,
            "payload": {
                "worker": {
                    "id": self.worker_id,
                    "modelId": model_id,
                    "maxTokens": 4096,
                    "capabilities": capabilities,
                    "address": f"localhost:{self.llm_proxy.endpoint.split(':')[-1]}",
                    "status": "ready",
                    "queueDepth": 0,
                },
                "authToken": self.worker_token,
            },
        }

        await self.ws.send(json.dumps(register_msg))
        console.print("[cyan]Registration sent, waiting for ACK...[/cyan]")

        # Wait for registration ACK
        try:
            response = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
            data = json.loads(response)

            if data.get("type") == "worker_ack":
                payload = data.get("payload", {})
                if payload.get("accepted"):
                    self.registered = True
                    console.print(f"[green]Registered as worker: {self.worker_id}[/green]")
                    return True
                else:
                    reason = payload.get("reason", "unknown")
                    console.print(f"[red]Registration rejected: {reason}[/red]")
                    return False
            elif data.get("type") == "error":
                error = data.get("payload", {})
                console.print(f"[red]Registration error: {error.get('message', 'unknown')}[/red]")
                return False
            else:
                console.print(f"[yellow]Unexpected response: {data.get('type')}[/yellow]")
                return False

        except asyncio.TimeoutError:
            console.print("[red]Registration timeout[/red]")
            return False

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to keep connection alive."""
        while self._running and self.connected and self.registered:
            try:
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                if self.ws and self.connected:
                    heartbeat = {
                        "id": self._generate_message_id(),
                        "type": "worker_heartbeat",
                        "timestamp": self._iso_timestamp(),
                        "senderId": self.worker_id,
                        "payload": {
                            "workerId": self.worker_id,
                            "status": "ready",
                            "queueDepth": 0,
                            "load": 0.0,
                        },
                    }
                    await self.ws.send(json.dumps(heartbeat))
                    logger.debug("Heartbeat sent")
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")
                break

    async def shutdown(self):
        """Request graceful shutdown."""
        self._shutdown_requested = True
        if self.ws:
            await self.ws.close()

    async def disconnect(self):
        """Close connection."""
        self._running = False
        self._shutdown_requested = True

        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        if self.ws and self.registered:
            # Send unregister message
            try:
                unregister = {
                    "id": self._generate_message_id(),
                    "type": "worker_unregister",
                    "timestamp": self._iso_timestamp(),
                    "senderId": self.worker_id,
                    "payload": {"workerId": self.worker_id},
                }
                await self.ws.send(json.dumps(unregister))
            except Exception:
                pass

        if self.ws:
            await self.ws.close()
            self.connected = False
            self.registered = False

        console.print("[yellow]Disconnected from backend[/yellow]")

    async def _handle_messages(self):
        """Handle incoming messages from backend."""
        try:
            async for message in self.ws:
                if self._shutdown_requested:
                    break
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                except json.JSONDecodeError:
                    console.print("[red]Invalid JSON received[/red]")
                except Exception as e:
                    logger.exception("Error processing message")
                    console.print(f"[red]Error processing message: {e}[/red]")

        except asyncio.CancelledError:
            logger.info("Message handler cancelled")
            raise
        except websockets.ConnectionClosed as e:
            console.print(f"[yellow]Connection closed: {e.reason}[/yellow]")
            self.connected = False
            self.registered = False

    async def _process_message(self, data: dict):
        """Process a message from the backend."""
        msg_type = data.get("type")
        trace_id = data.get("traceId", "")

        if msg_type == "intent_request":
            # Backend wants us to run an LLM query
            await self._handle_intent_request(data)

        elif msg_type == "nli_request":
            # Backend wants us to run NLI verification
            await self._handle_nli_request(data)

        elif msg_type == "nli_batch_request":
            # Backend wants us to run batch NLI verification
            await self._handle_nli_batch_request(data)

        elif msg_type == "ping":
            # Respond to keep-alive ping
            await self._send({
                "id": self._generate_message_id(),
                "type": "pong",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
            })

        elif msg_type == "intent_cancel":
            # Request to cancel in-flight request
            payload = data.get("payload", {})
            request_id = payload.get("requestId")
            console.print(f"[yellow]Cancel request: {request_id}[/yellow]")
            # TODO: Implement request cancellation

        elif msg_type == "error":
            payload = data.get("payload", {})
            console.print(f"[red]Backend error: {payload.get('message')}[/red]")

        elif msg_type == "server_notify":
            payload = data.get("payload", {})
            console.print(f"[cyan]Server notification: {payload}[/cyan]")

        else:
            logger.debug(f"Unknown message type: {msg_type}")

    async def _handle_intent_request(self, data: dict):
        """Handle an LLM inference request from the backend."""
        payload = data.get("payload", {})
        request_id = payload.get("requestId")
        trace_id = data.get("traceId", "")

        if not request_id:
            console.print("[red]Intent request missing requestId[/red]")
            return

        console.print(f"[cyan]Intent request {request_id[:8]}...[/cyan]")
        start_time = time.time()

        try:
            # Convert intent request to LLM request format
            llm_request = self._intent_to_llm_request(payload)

            # Execute on local LLM
            result = await self.llm_proxy.generate(llm_request)

            generation_ms = int((time.time() - start_time) * 1000)

            # Send response back to backend
            response = {
                "id": self._generate_message_id(),
                "type": "intent_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": "error" not in result or not result["error"],
                    "content": result.get("content", ""),
                    "tokensUsed": result.get("usage", {}).get("total_tokens", 0),
                    "generationMs": generation_ms,
                    "errorMessage": result.get("error", ""),
                },
            }

            await self._send(response)
            console.print(f"[green]Response sent for {request_id[:8]}... ({generation_ms}ms)[/green]")

            # Emit metrics callback
            if self.on_request_complete:
                tokens = result.get("usage", {}).get("completion_tokens", 0)
                # Extract NPC name from context if available
                npc_name = payload.get("context", {}).get("npcName", "NPC")
                # Time to first token (approximate - we don't have streaming)
                ttft_ms = generation_ms * 0.1  # Rough estimate: 10% of total time
                self.on_request_complete(npc_name, tokens, ttft_ms, generation_ms)

        except Exception as e:
            logger.exception("Intent handling error")
            console.print(f"[red]Intent error: {e}[/red]")

            # Send error response
            await self._send({
                "id": self._generate_message_id(),
                "type": "intent_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": False,
                    "content": "",
                    "errorMessage": str(e),
                },
            })

    async def _handle_nli_request(self, data: dict):
        """Handle an NLI verification request from the backend."""
        payload = data.get("payload", {})
        request_id = payload.get("requestId")
        trace_id = data.get("traceId", "")

        if not request_id:
            console.print("[red]NLI request missing requestId[/red]")
            return

        if not self.nli_service:
            await self._send({
                "id": self._generate_message_id(),
                "type": "nli_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": False,
                    "errorMessage": "NLI service not available",
                },
            })
            return

        claim = payload.get("claim", "")
        evidence = payload.get("evidence", "")

        console.print(f"[cyan]NLI request {request_id[:8]}...[/cyan]")
        start_time = time.time()

        try:
            result = self.nli_service.verify(claim, evidence)
            latency_ms = int((time.time() - start_time) * 1000)

            await self._send({
                "id": self._generate_message_id(),
                "type": "nli_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": True,
                    "entailment": result.entailment.value,
                    "confidence": result.confidence,
                    "probEntailment": result.prob_entailment,
                    "probNeutral": result.prob_neutral,
                    "probContradicts": result.prob_contradiction,
                    "latencyMs": latency_ms,
                },
            })
            console.print(f"[green]NLI response sent: {result.entailment.value} ({latency_ms}ms)[/green]")

        except Exception as e:
            logger.exception("NLI handling error")
            console.print(f"[red]NLI error: {e}[/red]")

            await self._send({
                "id": self._generate_message_id(),
                "type": "nli_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": False,
                    "errorMessage": str(e),
                },
            })

    async def _handle_nli_batch_request(self, data: dict):
        """Handle a batch NLI verification request from the backend."""
        payload = data.get("payload", {})
        request_id = payload.get("requestId")
        trace_id = data.get("traceId", "")

        if not request_id:
            console.print("[red]NLI batch request missing requestId[/red]")
            return

        if not self.nli_service:
            await self._send({
                "id": self._generate_message_id(),
                "type": "nli_batch_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": False,
                    "errorMessage": "NLI service not available",
                },
            })
            return

        claims = payload.get("claims", [])
        console.print(f"[cyan]NLI batch request {request_id[:8]}... ({len(claims)} pairs)[/cyan]")
        start_time = time.time()

        try:
            # Convert to list of tuples (claim, evidence)
            pairs = [(c.get("claim", ""), c.get("evidence", "")) for c in claims]
            results = self.nli_service.verify_batch(pairs)
            latency_ms = int((time.time() - start_time) * 1000)

            # Convert results to response format
            result_payloads = []
            for r in results:
                result_payloads.append({
                    "entailment": r.entailment.value,
                    "confidence": r.confidence,
                    "probEntailment": r.prob_entailment,
                    "probNeutral": r.prob_neutral,
                    "probContradicts": r.prob_contradiction,
                })

            await self._send({
                "id": self._generate_message_id(),
                "type": "nli_batch_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": True,
                    "results": result_payloads,
                    "latencyMs": latency_ms,
                },
            })
            console.print(f"[green]NLI batch response sent: {len(results)} results ({latency_ms}ms)[/green]")

        except Exception as e:
            logger.exception("NLI batch handling error")
            console.print(f"[red]NLI batch error: {e}[/red]")

            await self._send({
                "id": self._generate_message_id(),
                "type": "nli_batch_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": False,
                    "errorMessage": str(e),
                },
            })

    def _intent_to_llm_request(self, payload: dict) -> dict:
        """Convert backend IntentRequest to LLM request format."""
        context = payload.get("context", {})
        history = context.get("conversationHistory", [])

        # Build messages array
        messages = []

        # Add system prompt
        if payload.get("systemPrompt"):
            messages.append({
                "role": "system",
                "content": payload["systemPrompt"],
            })

        # Add conversation history
        for entry in history:
            messages.append({
                "role": entry.get("role", "user"),
                "content": entry.get("content", ""),
            })

        # Add current prompt as user message
        if payload.get("prompt"):
            messages.append({
                "role": "user",
                "content": payload["prompt"],
            })

        return {
            "messages": messages,
            "max_tokens": payload.get("maxTokens", 512),
            "temperature": payload.get("temperature", 0.7),
            "timeout": payload.get("timeoutMs", 120000) / 1000.0,  # Convert ms to seconds
        }

    async def _send(self, data: dict):
        """Send a message to the backend."""
        # Capture reference to avoid race condition
        ws = self.ws
        if not ws or not self.connected:
            logger.warning("Cannot send - not connected")
            return

        try:
            await ws.send(json.dumps(data))
        except Exception as e:
            logger.error(f"Send error: {e}")
            self.connected = False

    def _generate_message_id(self) -> str:
        """Generate a unique message ID."""
        timestamp = time.strftime("%Y%m%d%H%M%S")
        random_hex = secrets.token_hex(8)
        return f"{timestamp}-{random_hex}"

    def _iso_timestamp(self) -> str:
        """Get current time as ISO 8601 string."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
