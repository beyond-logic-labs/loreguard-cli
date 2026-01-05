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
        max_retries: int = -1,  # -1 = infinite retries, 0 = no retries (single try)
    ):
        self.backend_url = backend_url
        self.llm_proxy = llm_proxy
        self.worker_id = worker_id
        self.worker_token = worker_token
        self.model_id = model_id
        self.nli_service = nli_service
        self.log_callback = log_callback
        self.max_retries = max_retries

        self.ws: websockets.WebSocketClientProtocol | None = None
        self.connected = False
        self.registered = False
        self._reconnect_delay = 1  # Start with 1 second
        self._max_reconnect_delay = 60  # Max 60 seconds
        self._running = True
        self._shutdown_requested = False
        self._heartbeat_task: asyncio.Task | None = None
        self._retry_count = 0

        # Callback for metrics (called after each request completes)
        # Signature: (npc_name: str, tokens: int, ttft_ms: float, total_ms: float) -> None
        self.on_request_complete: Callable[[str, int, float, float], None] | None = None

        # Callback for verbose pass updates (called when backend sends pass_update)
        # Signature: (payload: dict) -> None
        self.on_pass_update: Callable[[dict], None] | None = None

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
            self._log("Error: Worker ID and API token are required", "error")
            self._log("Get an API token from loreguard.com dashboard", "warn")
            return

        last_error = ""

        while self._running:
            try:
                await self._connect_once()
                last_error = ""  # Clear on successful connection
            except websockets.exceptions.InvalidStatus as e:
                # Server rejected connection (e.g., 401, 500)
                logger.error("WebSocket connection rejected: HTTP %s", e.response.status_code)

                # Try to get error body
                error_body = ""
                try:
                    if hasattr(e.response, 'body') and e.response.body:
                        error_body = e.response.body.decode('utf-8', errors='ignore')
                    elif hasattr(e.response, 'read'):
                        error_body = e.response.read().decode('utf-8', errors='ignore')
                except Exception:
                    pass

                if error_body:
                    last_error = f"HTTP {e.response.status_code}: {error_body[:150]}"
                elif e.response.status_code == 401:
                    last_error = "HTTP 401: Invalid or expired token"
                elif e.response.status_code >= 500:
                    last_error = f"HTTP {e.response.status_code}: Server error"
                else:
                    last_error = f"HTTP {e.response.status_code}"

                self._log(last_error, "error")

            except websockets.exceptions.InvalidURI as e:
                logger.error("Invalid WebSocket URI: %s", e)
                last_error = f"Invalid backend URL: {e}"
                self._log(last_error, "error")
            except websockets.exceptions.InvalidHandshake as e:
                logger.error("WebSocket handshake failed: %s", e)
                last_error = f"Handshake failed: {e}"
                self._log(last_error, "error")
            except ConnectionRefusedError as e:
                logger.error("Connection refused: %s", e)
                last_error = "Connection refused - backend unreachable"
                self._log(last_error, "error")
            except OSError as e:
                logger.error("Network error: %s", e)
                last_error = f"Network error: {e}"
                self._log(last_error, "error")
            except Exception as e:
                logger.error("Connection error: %s (%s)", e, type(e).__name__)
                last_error = f"{type(e).__name__}: {e}"
                self._log(last_error, "error")

            if not self._running:
                break

            # Check retry limit
            self._retry_count += 1
            if self.max_retries >= 0 and self._retry_count > self.max_retries:
                # Show the actual error, not just "Connection failed"
                if last_error:
                    self._log(f"Connection failed: {last_error}", "error")
                else:
                    self._log("Connection failed", "error")
                self._running = False
                break

            # Exponential backoff for reconnection
            self._log(f"Reconnecting in {self._reconnect_delay}s... ({last_error})", "warn")
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)

    async def _connect_once(self):
        """Single connection attempt."""
        self._log(f"Connecting to {self.backend_url}...", "warn")

        self.ws = await websockets.connect(
            self.backend_url,
            additional_headers={"Authorization": f"Bearer {self.worker_token}"},
            ping_interval=30,
            ping_timeout=10,
        )
        self.connected = True
        connection_start = time.time()
        self._log("Connected to backend!", "success")

        # Register as worker
        success, error_reason = await self._register_worker()
        if not success:
            await self.ws.close()
            self.connected = False
            # Raise to trigger error handling in connect() with the reason
            raise Exception(f"Registration failed: {error_reason}")

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Start message handler
        try:
            await self._handle_messages()
        finally:
            # Only reset reconnect delay if we were connected for at least 30 seconds
            # This prevents rapid reconnection loops when connection drops immediately
            if time.time() - connection_start > 30:
                self._reconnect_delay = 1

            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass

    async def _register_worker(self) -> tuple[bool, str]:
        """Register as a worker with authentication.

        Returns:
            Tuple of (success, error_reason)
        """
        if not self.ws:
            return False, "No WebSocket connection"

        # Use the configured model_id (already mapped to backend-accepted ID)
        # Don't override with LLM server's model name (it returns filenames like Qwen3-4B-Instruct.gguf)
        model_id = self.model_id

        # Build capabilities list - add "nli" if NLI service is available
        capabilities = ["chat", "completion"]
        if self.nli_service is not None and self.nli_service.is_loaded:
            capabilities.append("nli")
            self._log("NLI capability enabled", "info")

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
        self._log("Registration sent, waiting for ACK...", "info")

        # Wait for registration ACK
        try:
            response = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
            data = json.loads(response)

            if data.get("type") == "worker_ack":
                payload = data.get("payload", {})
                if payload.get("accepted"):
                    self.registered = True
                    self._log(f"Registered as worker: {self.worker_id}", "success")
                    return True, ""
                else:
                    reason = payload.get("reason", "unknown")
                    self._log(f"Registration rejected: {reason}", "error")
                    return False, reason
            elif data.get("type") == "error":
                error = data.get("payload", {})
                msg = error.get("message", "unknown")
                self._log(f"Registration error: {msg}", "error")
                return False, msg
            else:
                msg = f"Unexpected response type: {data.get('type')}"
                self._log(msg, "warn")
                return False, msg

        except asyncio.TimeoutError:
            self._log("Registration timeout", "error")
            return False, "Registration timeout (no response from server)"

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

        self._log("Disconnected from backend", "warn")

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
                    self._log("Invalid JSON received", "error")
                except Exception as e:
                    logger.exception("Error processing message")
                    self._log(f"Error processing message: {e}", "error")

        except asyncio.CancelledError:
            logger.info("Message handler cancelled")
            raise
        except websockets.ConnectionClosed as e:
            self._log(f"Connection closed: {e.reason}", "warn")
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
            self._log(f"Cancel request: {request_id}", "warn")
            # TODO: Implement request cancellation

        elif msg_type == "error":
            payload = data.get("payload", {})
            self._log(f"Backend error: {payload.get('message')}", "error")

        elif msg_type == "server_notify":
            payload = data.get("payload", {})
            self._log(f"Server notification: {payload}", "info")

        elif msg_type == "pass_update":
            # Pipeline pass update (verbose mode)
            payload = data.get("payload", {})
            if self.on_pass_update:
                self.on_pass_update(payload)

        else:
            logger.debug(f"Unknown message type: {msg_type}")

    async def _handle_intent_request(self, data: dict):
        """Handle an LLM inference request from the backend."""
        payload = data.get("payload", {})
        request_id = payload.get("requestId")
        trace_id = data.get("traceId", "")

        if not request_id:
            self._log("Intent request missing requestId", "error")
            return

        self._log(f"Intent request {request_id[:8]}...", "info")
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
            self._log(f"Response sent for {request_id[:8]}... ({generation_ms}ms)", "success")

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
            self._log(f"Intent error: {e}", "error")

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
            self._log("NLI request missing requestId", "error")
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

        self._log(f"NLI request {request_id[:8]}...", "info")
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
            self._log(f"NLI response sent: {result.entailment.value} ({latency_ms}ms)", "success")

        except Exception as e:
            logger.exception("NLI handling error")
            self._log(f"NLI error: {e}", "error")

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
            self._log("NLI batch request missing requestId", "error")
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
        self._log(f"NLI batch request {request_id[:8]}... ({len(claims)} pairs)", "info")
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
            self._log(f"NLI batch response sent: {len(results)} results ({latency_ms}ms)", "success")

        except Exception as e:
            logger.exception("NLI batch handling error")
            self._log(f"NLI batch error: {e}", "error")

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
