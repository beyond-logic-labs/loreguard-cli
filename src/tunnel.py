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
from typing import Any, Callable, Optional, TYPE_CHECKING

import websockets
from rich.console import Console

from .llm import LLMProxy

if TYPE_CHECKING:
    from .nli import NLIService
    from .intent_classifier import IntentClassifier
    from .dialogue_act_classifier import DialogueActClassifier
    from .chunk_detector import ChunkDetector

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
        intent_classifier: Optional["IntentClassifier"] = None,
        dialogue_act_classifier: Optional["DialogueActClassifier"] = None,
        chunk_detector: Optional["ChunkDetector"] = None,
        log_callback: Callable[[str, str], None] | None = None,
        max_retries: int = -1,  # -1 = infinite retries, 0 = no retries (single try)
    ):
        self.backend_url = backend_url
        self.llm_proxy = llm_proxy
        self.worker_id = worker_id
        self.worker_token = worker_token
        self.model_id = model_id
        self.nli_service = nli_service
        self.intent_classifier = intent_classifier  # ADR-0010: Intent classification
        self.dialogue_act_classifier = dialogue_act_classifier  # Dialogue act classification for fillers
        self.chunk_detector = chunk_detector  # ADR-0023: Natural conversation breaks
        self.log_callback = log_callback
        self.max_retries = max_retries

        self.ws: websockets.WebSocketClientProtocol | None = None
        self.connected = False
        self.registered = False
        self.backend_version = ""  # Populated from worker_ack
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

        # Pending chat requests from local /api/chat endpoint
        # Maps request_id -> asyncio.Queue for routing responses
        self._pending_chat_requests: dict[str, asyncio.Queue[dict[str, Any]]] = {}

        # Follow-up cleanup tasks (ADR-0020)
        # Maps request_id -> cleanup task that removes queue after timeout
        self._follow_up_cleanup_tasks: dict[str, asyncio.Task] = {}
        self._follow_up_cleanup_delay = 20.0  # Keep queue alive for 20s after done

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
        if self.intent_classifier is not None and self.intent_classifier.is_loaded:
            capabilities.append("intent")
            self._log("Intent classification capability enabled (ADR-0010)", "info")
        if self.dialogue_act_classifier is not None and self.dialogue_act_classifier.is_loaded:
            capabilities.append("dialogue_act")
            self._log("Dialogue act capability enabled", "info")
        if self.chunk_detector is not None and self.chunk_detector.is_loaded:
            capabilities.append("chunk")
            self._log("Chunk detection capability enabled (ADR-0023)", "info")

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
                    # Capture backend version if provided
                    self.backend_version = payload.get("version", "")
                    self._log(f"Registered as worker: {self.worker_id}", "success")
                    if self.backend_version:
                        self._log(f"Backend version: {self.backend_version}", "info")
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

        elif msg_type == "intent_classify_request":
            # Backend wants us to classify user intent (ADR-0010)
            await self._handle_intent_classify_request(data)

        elif msg_type == "dialogue_act_classify_request":
            # Backend wants us to classify dialogue act
            await self._handle_dialogue_act_classify_request(data)

        elif msg_type == "promise_classify_request":
            # Backend wants us to detect follow-up promises (ADR-0020)
            await self._handle_promise_classify_request(data)

        elif msg_type == "chunk_detect_request":
            # Backend wants us to detect natural conversation breaks (ADR-0023)
            await self._handle_chunk_detect_request(data)

        elif msg_type == "nli_request":
            # Backend wants us to run NLI verification
            await self._handle_nli_request(data)

        elif msg_type == "nli_batch_request":
            # Backend wants us to run batch NLI verification
            await self._handle_nli_batch_request(data)

        elif msg_type == "llm_stream_request":
            # Backend wants streaming LLM generation
            await self._handle_llm_stream_request(data)

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

        elif msg_type == "chat_stream_token":
            # Token from backend for a chat request (local proxy architecture)
            await self._handle_chat_stream_token(data)

        elif msg_type == "chat_stream_done":
            # Chat response complete from backend (local proxy architecture)
            await self._handle_chat_stream_done(data)

        elif msg_type == "filler_message":
            # Immediate filler message for chat request
            await self._handle_filler_message(data)

        elif msg_type == "chat_follow_up":
            # Follow-up push from backend (ADR-0020)
            await self._handle_chat_follow_up(data)

        elif msg_type == "slot_save_request":
            # ADR-0014: Backend wants us to save KV cache to disk
            await self._handle_slot_save_request(data)

        elif msg_type == "slot_restore_request":
            # ADR-0014: Backend wants us to restore KV cache from disk
            await self._handle_slot_restore_request(data)

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

        # Log key request parameters for debugging
        force_json = payload.get("forceJsonOutput", False)
        logger.debug(f"Intent request {request_id[:8]}: forceJson={force_json}")

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

    async def _handle_llm_stream_request(self, data: dict):
        """Handle a streaming LLM inference request from the backend.

        Streams tokens back to the backend via llm_token messages,
        then sends llm_stream_complete when done.
        """
        payload = data.get("payload", {})
        request_id = payload.get("requestId")
        trace_id = data.get("traceId", "")

        if not request_id:
            self._log("LLM stream request missing requestId", "error")
            return

        self._log(f"LLM stream request {request_id[:8]}...", "info")
        start_time = time.time()

        # Log key payload fields for debugging streaming issues
        force_json = payload.get("forceJsonOutput", False)
        max_tokens = payload.get("maxTokens", 512)
        has_schema = bool(payload.get("jsonSchema"))
        logger.debug(f"Stream request {request_id[:8]}: forceJson={force_json}, schema={has_schema}, maxTokens={max_tokens}")

        try:
            # Convert stream request payload to LLM request format
            # ADR-0014: Use cognitiveContext for KV cache sharing if provided
            # cognitiveContext contains static NPC content (personality, memory, voice, etc.)
            # that is identical across all pipeline passes, enabling llama-server
            # to cache and reuse the KV prefix.
            # Note: Backend sends "cognitiveContext" (messages.go), falls back to "systemPrompt"
            cognitive_context = payload.get("cognitiveContext") or payload.get("systemPrompt", "")

            # Debug: Log cognitive context presence and voice tag
            if cognitive_context:
                has_voice = "<voice>" in cognitive_context
                logger.info(f"[LLM Stream] cognitiveContext received ({len(cognitive_context)} chars, has_voice={has_voice})")
                if has_voice:
                    # Extract and log voice content for debugging
                    import re
                    voice_match = re.search(r'<voice>(.*?)</voice>', cognitive_context, re.DOTALL)
                    if voice_match:
                        logger.info(f"[LLM Stream] Voice rules: {voice_match.group(1)[:100]}")
            else:
                logger.warning("[LLM Stream] No cognitiveContext received - style/voice will not be applied!")

            # Build messages array with conversation history
            messages = []

            # Add system message (cognitive context)
            if cognitive_context:
                messages.append({"role": "system", "content": cognitive_context})

            # Add conversation history (player/NPC previous exchanges)
            # This is sent by the backend in worker_llm_client.go
            conversation_history = payload.get("conversationHistory", [])
            for entry in conversation_history:
                messages.append({
                    "role": entry.get("role", "user"),
                    "content": entry.get("content", ""),
                })

            # Add current user message (pass instructions + dynamic content)
            user_message = payload.get("userMessage", "")
            if user_message:
                messages.append({"role": "user", "content": user_message})

            llm_request = {
                "messages": messages,
                "max_tokens": payload.get("maxTokens", 512),
                "temperature": payload.get("temperature", 0.7),
                "timeout": payload.get("timeoutMs", 120000) / 1000,  # Convert to seconds
                "cache_prompt": payload.get("cacheEnabled", False),  # Enable KV cache prefix sharing
            }

            # Pass through JSON output settings (ADR-0012: grammar-constrained JSON)
            if payload.get("forceJsonOutput"):
                llm_request["force_json"] = True
                if payload.get("jsonSchema"):
                    llm_request["json_schema"] = payload["jsonSchema"]

            # Pass through thinking mode control (important for JSON output)
            if payload.get("disableThinking"):
                llm_request["disable_thinking"] = True

            logger.debug(f"LLM request config: force_json={llm_request.get('force_json', False)}, max_tokens={llm_request.get('max_tokens')}")

            # Stream tokens from LLM
            token_count = 0
            content_parts = []
            thinking = ""
            usage = {}

            # Diagnostic: track inter-token timing to identify stalls
            last_token_time = time.time()
            first_token_time = None

            async for chunk in self.llm_proxy.generate_streaming(llm_request):
                chunk_type = chunk.get("type")

                if chunk_type == "token":
                    token = chunk.get("token", "")
                    if token:
                        content_parts.append(token)
                        token_count += 1

                        # Track time to first token
                        if token_count == 1:
                            first_token_time = time.time()
                            logger.debug(f"[Stream {request_id[:8]}] First token received")

                        # Diagnostic: check for slow token generation
                        now = time.time()
                        token_gap = now - last_token_time
                        if token_gap > 1.0:
                            # Large gap between tokens - potential stall indicator
                            logger.warning(
                                f"[Stream {request_id[:8]}] Token gap: {token_gap:.2f}s after {token_count} tokens"
                            )
                        last_token_time = now

                        # Send token message to backend
                        send_start = time.time()
                        await self._send({
                            "id": self._generate_message_id(),
                            "type": "llm_token",
                            "timestamp": self._iso_timestamp(),
                            "senderId": self.worker_id,
                            "traceId": trace_id,
                            "payload": {
                                "requestId": request_id,
                                "token": token,
                                "index": chunk.get("index", token_count - 1),
                            },
                        })
                        send_elapsed = time.time() - send_start
                        if send_elapsed > 0.5:
                            # Slow WebSocket send - potential backpressure
                            logger.warning(
                                f"[Stream {request_id[:8]}] Slow WebSocket send: {send_elapsed:.2f}s at token {token_count}"
                            )

                elif chunk_type == "done":
                    # Final chunk with complete content
                    thinking = chunk.get("thinking", "")
                    usage = chunk.get("usage", {})
                    # Use the processed content from the done chunk
                    final_content = chunk.get("content", "".join(content_parts))

                    latency_ms = int((time.time() - start_time) * 1000)

                    logger.debug(f"[Stream {request_id[:8]}] Sending llm_stream_complete: {token_count} tokens, {len(final_content)} chars")

                    # Send completion message
                    await self._send({
                        "id": self._generate_message_id(),
                        "type": "llm_stream_complete",
                        "timestamp": self._iso_timestamp(),
                        "senderId": self.worker_id,
                        "traceId": trace_id,
                        "payload": {
                            "requestId": request_id,
                            "workerId": self.worker_id,
                            "success": True,
                            "content": final_content,
                            "thinking": thinking,
                            "tokenCount": token_count,
                            "latencyMs": latency_ms,
                        },
                    })

                    self._log(
                        f"Stream complete for {request_id[:8]}... ({token_count} tokens, {latency_ms}ms)",
                        "success",
                    )

                    # Emit metrics callback
                    if self.on_request_complete:
                        npc_name = payload.get("context", {}).get("npcName", "NPC")
                        # Estimate TTFT as time for first token (actual TTFT would require timing first token)
                        ttft_ms = latency_ms / max(token_count, 1)
                        self.on_request_complete(npc_name, token_count, ttft_ms, latency_ms)

                    return  # Done

                elif chunk_type == "error":
                    # Error during streaming
                    error_msg = chunk.get("error", "Unknown error")
                    latency_ms = int((time.time() - start_time) * 1000)

                    await self._send({
                        "id": self._generate_message_id(),
                        "type": "llm_stream_complete",
                        "timestamp": self._iso_timestamp(),
                        "senderId": self.worker_id,
                        "traceId": trace_id,
                        "payload": {
                            "requestId": request_id,
                            "workerId": self.worker_id,
                            "success": False,
                            "content": "",
                            "tokenCount": token_count,
                            "latencyMs": latency_ms,
                            "errorMessage": error_msg,
                        },
                    })

                    self._log(f"Stream error for {request_id[:8]}...: {error_msg}", "error")
                    return

            # Safety net: if generator exits without yielding "done" or "error",
            # send completion with whatever content we have (shouldn't happen normally)
            latency_ms = int((time.time() - start_time) * 1000)
            self._log(f"Stream ended unexpectedly for {request_id[:8]}... ({token_count} tokens)", "warn")
            await self._send({
                "id": self._generate_message_id(),
                "type": "llm_stream_complete",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": token_count > 0,  # Partial success if we got tokens
                    "content": "".join(content_parts),
                    "tokenCount": token_count,
                    "latencyMs": latency_ms,
                    "errorMessage": "Stream ended without completion signal" if token_count == 0 else "",
                },
            })

        except Exception as e:
            logger.exception("LLM stream handling error")
            self._log(f"Stream error: {e}", "error")

            latency_ms = int((time.time() - start_time) * 1000)

            # Send error completion
            await self._send({
                "id": self._generate_message_id(),
                "type": "llm_stream_complete",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": False,
                    "content": "",
                    "tokenCount": 0,
                    "latencyMs": latency_ms,
                    "errorMessage": str(e),
                },
            })

    async def _handle_intent_classify_request(self, data: dict):
        """Handle an intent classification request from the backend (ADR-0010)."""
        payload = data.get("payload", {})
        request_id = payload.get("requestId")
        trace_id = data.get("traceId", "")

        if not request_id:
            self._log("Intent classify request missing requestId", "error")
            return

        if not self.intent_classifier:
            await self._send({
                "id": self._generate_message_id(),
                "type": "intent_classify_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": False,
                    "errorMessage": "Intent classifier not available",
                },
            })
            return

        query = payload.get("query", "")
        timeout_ms = payload.get("timeoutMs", 5000)  # Default 5s, allows cold start model loading
        timeout_s = timeout_ms / 1000.0

        self._log(f"Intent classify request {request_id[:8]}...", "info")
        start_time = time.time()

        try:
            # Run classification with timeout to avoid blocking WS loop
            result = await asyncio.wait_for(
                asyncio.to_thread(self.intent_classifier.classify, query),
                timeout=timeout_s
            )
            latency_ms = int((time.time() - start_time) * 1000)

            await self._send({
                "id": self._generate_message_id(),
                "type": "intent_classify_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": True,
                    "intent": result.intent.value,
                    "confidence": result.confidence,
                    "latencyMs": latency_ms,
                },
            })
            self._log(f"Intent classify response: {result.intent.value} ({latency_ms}ms)", "success")

        except asyncio.TimeoutError:
            latency_ms = int((time.time() - start_time) * 1000)
            self._log(f"Intent classify timeout after {latency_ms}ms (limit: {timeout_ms}ms)", "warning")

            await self._send({
                "id": self._generate_message_id(),
                "type": "intent_classify_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": False,
                    "errorMessage": f"Classification timeout ({timeout_ms}ms)",
                },
            })

        except Exception as e:
            logger.exception("Intent classification error")
            self._log(f"Intent classify error: {e}", "error")

            await self._send({
                "id": self._generate_message_id(),
                "type": "intent_classify_response",
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

    async def _handle_dialogue_act_classify_request(self, data: dict):
        """Handle a dialogue act classification request from the backend."""
        payload = data.get("payload", {})
        request_id = payload.get("requestId")
        trace_id = data.get("traceId", "")

        if not request_id:
            self._log("Dialogue act request missing requestId", "error")
            return

        if not self.dialogue_act_classifier:
            await self._send({
                "id": self._generate_message_id(),
                "type": "dialogue_act_classify_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": False,
                    "errorMessage": "Dialogue act classifier not available",
                },
            })
            return

        query = payload.get("query", "")
        timeout_ms = payload.get("timeoutMs", 5000)
        timeout_s = timeout_ms / 1000.0

        self._log(f"Dialogue act request {request_id[:8]}...", "info")
        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self.dialogue_act_classifier.classify, query),
                timeout=timeout_s
            )
            latency_ms = int((time.time() - start_time) * 1000)

            await self._send({
                "id": self._generate_message_id(),
                "type": "dialogue_act_classify_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": True,
                    "dialogueAct": result.dialogue_act,
                    "confidence": result.confidence,
                    "latencyMs": latency_ms,
                },
            })
            self._log(f"Dialogue act response: {result.dialogue_act} ({latency_ms}ms)", "success")

        except asyncio.TimeoutError:
            latency_ms = int((time.time() - start_time) * 1000)
            self._log(f"Dialogue act timeout after {latency_ms}ms (limit: {timeout_ms}ms)", "warning")

            await self._send({
                "id": self._generate_message_id(),
                "type": "dialogue_act_classify_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": False,
                    "errorMessage": f"Classification timeout ({timeout_ms}ms)",
                },
            })

        except Exception as e:
            logger.exception("Dialogue act classification error")
            self._log(f"Dialogue act error: {e}", "error")

            await self._send({
                "id": self._generate_message_id(),
                "type": "dialogue_act_classify_response",
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

    async def _handle_promise_classify_request(self, data: dict):
        """Handle a promise classification request from the backend (ADR-0020).

        Uses the same IntentClassifier (DeBERTa) with a promise-specific hypothesis
        to detect if an NPC response contains a follow-up promise.
        """
        payload = data.get("payload", {})
        request_id = payload.get("requestId")
        trace_id = data.get("traceId", "")

        if not request_id:
            self._log("Promise classify request missing requestId", "error")
            return

        if not self.intent_classifier:
            await self._send({
                "id": self._generate_message_id(),
                "type": "promise_classify_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": False,
                    "errorMessage": "Intent classifier not available (required for promise detection)",
                },
            })
            return

        text = payload.get("text", "")
        timeout_ms = payload.get("timeoutMs", 3000)  # Default 3s for promise detection
        timeout_s = timeout_ms / 1000.0

        self._log(f"Promise classify request {request_id[:8]}...", "info")
        start_time = time.time()

        try:
            # Run promise classification with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(self.intent_classifier.classify_promise, text),
                timeout=timeout_s
            )
            latency_ms = int((time.time() - start_time) * 1000)

            await self._send({
                "id": self._generate_message_id(),
                "type": "promise_classify_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": True,
                    "hasPromise": result.has_promise,
                    "confidence": result.confidence,
                    "latencyMs": latency_ms,
                },
            })
            self._log(f"Promise classify response: has_promise={result.has_promise} ({latency_ms}ms)", "success")

        except asyncio.TimeoutError:
            latency_ms = int((time.time() - start_time) * 1000)
            self._log(f"Promise classify timeout after {latency_ms}ms (limit: {timeout_ms}ms)", "warning")

            await self._send({
                "id": self._generate_message_id(),
                "type": "promise_classify_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": False,
                    "errorMessage": f"Classification timeout ({timeout_ms}ms)",
                },
            })

        except Exception as e:
            logger.exception("Promise classification error")
            self._log(f"Promise classify error: {e}", "error")

            await self._send({
                "id": self._generate_message_id(),
                "type": "promise_classify_response",
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

    async def _handle_chunk_detect_request(self, data: dict):
        """Handle a chunk detection request from the backend (ADR-0023).

        Uses DeBERTa to detect natural conversation breaks in NPC responses.
        Returns the text split into chunks for sequential delivery.
        """
        payload = data.get("payload", {})
        request_id = payload.get("requestId")
        trace_id = data.get("traceId", "")

        if not request_id:
            self._log("Chunk detect request missing requestId", "error")
            return

        if not self.chunk_detector:
            await self._send({
                "id": self._generate_message_id(),
                "type": "chunk_detect_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": False,
                    "errorMessage": "Chunk detector not available",
                },
            })
            return

        text = payload.get("text", "")
        timeout_ms = payload.get("timeoutMs", 3000)  # Default 3s for chunk detection
        timeout_s = timeout_ms / 1000.0

        self._log(f"Chunk detect request {request_id[:8]}... ({len(text)} chars)", "info")
        start_time = time.time()

        try:
            # Run chunk detection with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(self.chunk_detector.detect, text),
                timeout=timeout_s
            )
            latency_ms = int((time.time() - start_time) * 1000)

            # Convert TextChunk objects to dicts for JSON serialization
            chunks = [{"text": chunk.text, "index": chunk.index} for chunk in result.chunks]

            await self._send({
                "id": self._generate_message_id(),
                "type": "chunk_detect_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": True,
                    "chunks": chunks,
                    "latencyMs": latency_ms,
                },
            })
            self._log(f"Chunk detect response: {len(chunks)} chunks ({latency_ms}ms)", "success")

        except asyncio.TimeoutError:
            latency_ms = int((time.time() - start_time) * 1000)
            self._log(f"Chunk detect timeout after {latency_ms}ms (limit: {timeout_ms}ms)", "warning")

            await self._send({
                "id": self._generate_message_id(),
                "type": "chunk_detect_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "workerId": self.worker_id,
                    "success": False,
                    "errorMessage": f"Chunk detection timeout ({timeout_ms}ms)",
                },
            })

        except Exception as e:
            logger.exception("Chunk detection error")
            self._log(f"Chunk detect error: {e}", "error")

            await self._send({
                "id": self._generate_message_id(),
                "type": "chunk_detect_response",
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

    # =========================================================================
    # Local Proxy Architecture: Chat Request/Response Handling
    # =========================================================================

    async def send_chat_request(
        self,
        request_id: str,
        character_id: str,
        message: str,
        player_id: str = "",
        player_handle: str = "",
        current_context: str = "",
        scenario_id: str = "",
        history: list[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
        verbose: bool = False,
        api_token: str = "",
        max_speech_tokens: int = 0,
    ) -> asyncio.Queue[dict[str, Any]]:
        """Send a chat request to the backend and return a queue for responses.

        The game client calls POST /api/chat on localhost, which calls this method.
        The backend runs the pipeline and sends tokens/completion back.

        Args:
            player_id: Player's unique ID for per-player state/memory. If empty,
                      backend uses the owner ID (developer mode fallback).

        Returns:
            Queue that will receive:
            - {"type": "token", "token": "...", "index": N} for each token
            - {"type": "done", "data": {...}} when complete
            - {"type": "error", "error": "..."} on error
        """
        if not self.connected or not self.registered:
            # Return a queue with error
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            await queue.put({"type": "error", "error": "Not connected to backend"})
            return queue

        # Create response queue
        queue = asyncio.Queue()
        self._pending_chat_requests[request_id] = queue

        # Send chat_request to backend
        payload = {
            "requestId": request_id,
            "characterId": character_id,
            "message": message,
            "playerHandle": player_handle,
            "currentContext": current_context,
            "scenarioId": scenario_id,
            "history": history or [],
            "enableThinking": enable_thinking,
            "verbose": verbose,
            "stream": True,  # Always streaming for SSE
            "apiToken": api_token,  # For backend to authorize character access
        }
        # Only include playerId if provided
        if player_id:
            payload["playerId"] = player_id
        # Only include maxSpeechTokens if explicitly set (non-zero)
        if max_speech_tokens > 0:
            payload["maxSpeechTokens"] = max_speech_tokens

        await self._send({
            "id": self._generate_message_id(),
            "type": "chat_request",
            "timestamp": self._iso_timestamp(),
            "senderId": self.worker_id,
            "payload": payload,
        })

        self._log(f"Chat request sent: {request_id[:8]}... (character: {character_id})", "info")
        return queue

    def cancel_chat_request(self, request_id: str):
        """Remove a pending chat request (e.g., on client disconnect)."""
        self._pending_chat_requests.pop(request_id, None)

    async def _handle_chat_stream_token(self, data: dict):
        """Handle a chat_stream_token message from the backend."""
        payload = data.get("payload", {})
        request_id = payload.get("requestId")

        if not request_id:
            logger.warning("chat_stream_token missing requestId")
            return

        queue = self._pending_chat_requests.get(request_id)
        if not queue:
            logger.debug(f"No pending chat request for {request_id[:8]}...")
            return

        # Put token in queue
        await queue.put({
            "type": "token",
            "token": payload.get("t", ""),
            "index": payload.get("index", 0),
        })

    async def _handle_chat_stream_done(self, data: dict):
        """Handle a chat_stream_done message from the backend."""
        payload = data.get("payload", {})
        request_id = payload.get("requestId")

        if not request_id:
            logger.warning("chat_stream_done missing requestId")
            return

        queue = self._pending_chat_requests.get(request_id)
        if not queue:
            logger.debug(f"No pending chat request for {request_id[:8]}...")
            return

        # Check for error (on error, remove the queue immediately)
        if payload.get("errorMessage"):
            self._pending_chat_requests.pop(request_id, None)
            await queue.put({
                "type": "error",
                "error": payload.get("errorMessage"),
            })
        else:
            # Put completion in queue
            await queue.put({
                "type": "done",
                "data": {
                    "speech": payload.get("speech", ""),
                    "thoughts": payload.get("thoughts", ""),
                    "citations": payload.get("citations", []),
                    "verified": payload.get("verified", False),
                    "retries": payload.get("retries", 0),
                    "latency_ms": payload.get("latencyMs", 0),
                },
            })
            # Schedule cleanup after delay to allow follow-ups (ADR-0020)
            self._schedule_follow_up_cleanup(request_id)

        self._log(
            f"Chat response received: {request_id[:8]}... "
            f"(verified: {payload.get('verified', False)})",
            "success" if payload.get("verified") else "warn",
        )

    def _schedule_follow_up_cleanup(self, request_id: str):
        """Schedule cleanup of pending chat request after follow-up timeout."""
        # Cancel existing cleanup task if any
        existing = self._follow_up_cleanup_tasks.get(request_id)
        if existing:
            existing.cancel()
        # Create new cleanup task
        cleanup_task = asyncio.create_task(self._run_follow_up_cleanup(request_id))
        self._follow_up_cleanup_tasks[request_id] = cleanup_task

    async def _run_follow_up_cleanup(self, request_id: str):
        """Wait for follow-up timeout then cleanup the queue."""
        try:
            await asyncio.sleep(self._follow_up_cleanup_delay)
        except asyncio.CancelledError:
            return
        # Cleanup queue and task
        self._pending_chat_requests.pop(request_id, None)
        self._follow_up_cleanup_tasks.pop(request_id, None)

    async def _handle_filler_message(self, data: dict):
        """Handle a filler_message from the backend."""
        payload = data.get("payload", {})
        request_id = payload.get("requestId")

        if not request_id:
            logger.warning("filler_message missing requestId")
            return

        queue = self._pending_chat_requests.get(request_id)
        if not queue:
            logger.debug(f"No pending chat request for filler {request_id[:8]}...")
            return

        await queue.put({
            "type": "filler",
            "data": {
                "text": payload.get("text", ""),
                "dialogueAct": payload.get("dialogueAct", ""),
            },
        })

    async def _handle_chat_follow_up(self, data: dict):
        """Handle a chat_follow_up push from the backend (ADR-0020)."""
        payload = data.get("payload", {})
        request_id = payload.get("requestId")

        if not request_id:
            logger.warning("chat_follow_up missing requestId")
            return

        queue = self._pending_chat_requests.get(request_id)
        if not queue:
            logger.debug(f"No pending chat request for follow-up {request_id[:8]}...")
            return

        follow_up = {
            "speech": payload.get("speech", ""),
        }
        delay_ms = payload.get("delay_ms")
        if isinstance(delay_ms, int):
            follow_up["delay_ms"] = delay_ms

        await queue.put({
            "type": "follow_up",
            "data": follow_up,
        })
        # Reschedule cleanup to allow more follow-ups
        self._schedule_follow_up_cleanup(request_id)
        self._log(
            f"Follow-up received: {request_id[:8]}... "
            f"(delay: {delay_ms or 0}ms)",
            "info",
        )

    def _intent_to_llm_request(self, payload: dict) -> dict:
        """Convert backend IntentRequest to LLM request format."""
        context = payload.get("context", {})
        history = context.get("conversationHistory", [])

        # Build messages array
        messages = []

        # ADR-0014: Use cognitiveContext for KV cache sharing if provided
        # cognitiveContext contains static NPC content (personality, memory, voice, etc.)
        # Note: Backend sends "cognitiveContext" (messages.go), falls back to "systemPrompt"
        cognitive_context = payload.get("cognitiveContext") or payload.get("systemPrompt", "")

        # Add system prompt (cognitiveContext takes priority)
        if cognitive_context:
            messages.append({
                "role": "system",
                "content": cognitive_context,
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

        request = {
            "messages": messages,
            "max_tokens": payload.get("maxTokens", 512),
            "temperature": payload.get("temperature", 0.7),
            "timeout": payload.get("timeoutMs", 120000) / 1000.0,  # Convert ms to seconds
            "cache_prompt": payload.get("cacheEnabled", False),  # ADR-0014: KV cache prefix sharing
        }

        # Pass through JSON output settings (ADR-0012: grammar-constrained JSON)
        if payload.get("forceJsonOutput"):
            logger.info(f"Intent request has forceJsonOutput=True, jsonSchema={'yes' if payload.get('jsonSchema') else 'no'}")
            request["force_json"] = True
            if payload.get("jsonSchema"):
                request["json_schema"] = payload["jsonSchema"]

        # Pass through thinking mode control (important for JSON output)
        if payload.get("disableThinking"):
            request["disable_thinking"] = True

        return request

    def _validate_slot_filename(self, filename: str) -> tuple[bool, str]:
        """Validate a slot cache filename for security.

        Returns (is_valid, error_message). Mirrors Go's sanitizeFilenameComponent.
        Security: Prevents path traversal and injection attacks.
        """
        import re

        if not filename:
            return False, "filename is required"

        # Check length (match Go's 64 char limit + extension)
        if len(filename) > 80:
            return False, "filename exceeds maximum length"

        # Check for path traversal attempts
        if ".." in filename or "/" in filename or "\\" in filename:
            return False, "invalid filename: path traversal not allowed"

        # Validate filename format: npc_{id}_{hash}.bin
        # Only allow alphanumeric, underscore, hyphen, and .bin extension
        if not re.match(r'^[a-zA-Z0-9_-]+\.bin$', filename):
            return False, "invalid filename format"

        return True, ""

    async def _handle_slot_save_request(self, data: dict):
        """Handle a slot save request from the backend (ADR-0014).

        Saves the KV cache for a slot to disk, enabling cross-message caching
        of warmup context (NPC personality, memories, etc.).
        """
        payload = data.get("payload", {})
        request_id = payload.get("requestId")
        trace_id = data.get("traceId", "")
        slot_id = payload.get("slotId", 0)
        filename = payload.get("filename", "")

        if not request_id:
            self._log("Slot save request missing requestId", "error")
            return

        # Security: Validate slot ID range (0-7, matching Go validation)
        if not isinstance(slot_id, int) or slot_id < 0 or slot_id > 7:
            await self._send({
                "id": self._generate_message_id(),
                "type": "slot_save_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "success": False,
                    "errorMessage": f"invalid slotId {slot_id}: must be 0-7",
                },
            })
            return

        # Security: Validate filename to prevent path traversal
        is_valid, error_msg = self._validate_slot_filename(filename)
        if not is_valid:
            await self._send({
                "id": self._generate_message_id(),
                "type": "slot_save_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "success": False,
                    "errorMessage": error_msg,
                },
            })
            return

        self._log(f"Slot save request {request_id[:8]}... (slot={slot_id}, file={filename})", "info")
        start_time = time.time()

        try:
            result = await self.llm_proxy.save_slot(slot_id, filename)
            latency_ms = int((time.time() - start_time) * 1000)

            if "error" in result:
                await self._send({
                    "id": self._generate_message_id(),
                    "type": "slot_save_response",
                    "timestamp": self._iso_timestamp(),
                    "senderId": self.worker_id,
                    "traceId": trace_id,
                    "payload": {
                        "requestId": request_id,
                        "success": False,
                        "errorMessage": result["error"],
                        "latencyMs": latency_ms,
                    },
                })
            else:
                await self._send({
                    "id": self._generate_message_id(),
                    "type": "slot_save_response",
                    "timestamp": self._iso_timestamp(),
                    "senderId": self.worker_id,
                    "traceId": trace_id,
                    "payload": {
                        "requestId": request_id,
                        "success": True,
                        "tokensSaved": result.get("n_saved", 0),
                        "bytesWritten": result.get("n_written", 0),
                        "latencyMs": latency_ms,
                    },
                })
                self._log(f"Slot saved: {result.get('n_saved', 0)} tokens ({latency_ms}ms)", "success")

        except Exception as e:
            logger.exception("Slot save handling error")
            self._log(f"Slot save error: {e}", "error")

            await self._send({
                "id": self._generate_message_id(),
                "type": "slot_save_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "success": False,
                    "errorMessage": str(e),
                },
            })

    async def _handle_slot_restore_request(self, data: dict):
        """Handle a slot restore request from the backend (ADR-0014).

        Restores the KV cache for a slot from disk, warming up the model
        with previously cached NPC context.
        """
        payload = data.get("payload", {})
        request_id = payload.get("requestId")
        trace_id = data.get("traceId", "")
        slot_id = payload.get("slotId", 0)
        filename = payload.get("filename", "")

        if not request_id:
            self._log("Slot restore request missing requestId", "error")
            return

        # Security: Validate slot ID range (0-7, matching Go validation)
        if not isinstance(slot_id, int) or slot_id < 0 or slot_id > 7:
            await self._send({
                "id": self._generate_message_id(),
                "type": "slot_restore_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "success": False,
                    "errorMessage": f"invalid slotId {slot_id}: must be 0-7",
                },
            })
            return

        # Security: Validate filename to prevent path traversal
        is_valid, error_msg = self._validate_slot_filename(filename)
        if not is_valid:
            await self._send({
                "id": self._generate_message_id(),
                "type": "slot_restore_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "success": False,
                    "errorMessage": error_msg,
                },
            })
            return

        self._log(f"Slot restore request {request_id[:8]}... (slot={slot_id}, file={filename})", "info")
        start_time = time.time()

        try:
            result = await self.llm_proxy.restore_slot(slot_id, filename)
            latency_ms = int((time.time() - start_time) * 1000)

            if "error" in result:
                await self._send({
                    "id": self._generate_message_id(),
                    "type": "slot_restore_response",
                    "timestamp": self._iso_timestamp(),
                    "senderId": self.worker_id,
                    "traceId": trace_id,
                    "payload": {
                        "requestId": request_id,
                        "success": False,
                        "errorMessage": result["error"],
                        "latencyMs": latency_ms,
                    },
                })
            else:
                cold_start = result.get("cold_start", False)
                await self._send({
                    "id": self._generate_message_id(),
                    "type": "slot_restore_response",
                    "timestamp": self._iso_timestamp(),
                    "senderId": self.worker_id,
                    "traceId": trace_id,
                    "payload": {
                        "requestId": request_id,
                        "success": True,
                        "coldStart": cold_start,
                        "tokensRestored": result.get("n_restored", 0),
                        "bytesRead": result.get("n_read", 0),
                        "latencyMs": latency_ms,
                    },
                })
                if cold_start:
                    self._log(f"Slot restore: cold start (no cache file) ({latency_ms}ms)", "info")
                else:
                    self._log(f"Slot restored: {result.get('n_restored', 0)} tokens ({latency_ms}ms)", "success")

        except Exception as e:
            logger.exception("Slot restore handling error")
            self._log(f"Slot restore error: {e}", "error")

            await self._send({
                "id": self._generate_message_id(),
                "type": "slot_restore_response",
                "timestamp": self._iso_timestamp(),
                "senderId": self.worker_id,
                "traceId": trace_id,
                "payload": {
                    "requestId": request_id,
                    "success": False,
                    "errorMessage": str(e),
                },
            })

    async def _send(self, data: dict, timeout: float = 5.0):
        """Send a message to the backend with timeout.

        Args:
            data: Message to send as dict
            timeout: Maximum seconds to wait for send (default 5s)
                     This prevents indefinite hangs if WebSocket is blocked
                     (e.g., Cloudflare tunnel backpressure)
        """
        # Capture reference to avoid race condition
        ws = self.ws
        if not ws or not self.connected:
            logger.warning("Cannot send - not connected")
            return

        try:
            # Add timeout to prevent indefinite hangs on blocked WebSocket
            # This is critical for streaming - if send blocks, we stop reading
            # from llama-server and the entire pipeline stalls
            await asyncio.wait_for(ws.send(json.dumps(data)), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"WebSocket send timed out after {timeout}s - connection may be blocked")
            self.connected = False
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
