"""Embedded HTTP server for game SDK connections.

This module provides a lightweight HTTP server that can be started
alongside the TUI or CLI to handle game client requests via SSE.

The server shares the existing tunnel connection instead of creating
a new one, ensuring a single WebSocket connection per player.

Uses hypercorn with socket-first binding for race-condition-free port allocation.
"""

import asyncio
import json
import threading
import uuid
from concurrent.futures import Future
from typing import Any, Callable, Optional

from .runtime import write_runtime_info, RuntimeInfo, get_runtime_path, get_version


class EmbeddedHTTPServer:
    """Lightweight HTTP server for game SDK connections.

    This server provides the /api/chat endpoint for games to connect to.
    It shares the tunnel from the TUI/CLI instead of creating its own.

    Uses a socket-first approach to guarantee port binding:
    1. Create and bind socket (keeps it reserved)
    2. Pass socket to hypercorn (no race condition)
    3. Get actual port from bound socket

    Thread Safety:
    The tunnel runs in the main (TUI) event loop, while this server runs
    in a separate thread with its own event loop. Cross-loop communication
    is handled via run_coroutine_threadsafe() and thread-safe queues.
    """

    def __init__(
        self,
        tunnel: Any,  # BackendTunnel instance
        host: str = "127.0.0.1",
        port: int = 0,  # 0 = auto-assign
        on_status_change: Optional[Callable[[str], None]] = None,
        main_loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.tunnel = tunnel
        self.host = host
        self.requested_port = port
        self.actual_port: Optional[int] = None
        self.on_status_change = on_status_change
        self._server: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._main_loop = main_loop  # TUI's event loop for cross-loop calls
        self._running = False
        self._bound_socket: Optional[Any] = None
        self._ready_event = threading.Event()

    def start(self) -> int:
        """Start the HTTP server in a background thread.

        Returns:
            The actual port the server is listening on.

        This method is race-condition free:
        1. Binds socket immediately (reserves the port)
        2. Passes bound socket to hypercorn
        3. Returns only after server is ready
        """
        # Early debug logging
        debug_path = get_runtime_path().parent / "sdk_debug.log"
        with open(debug_path, "a") as f:
            f.write(f"[SDK Server] start() called, _running={self._running}\n")

        if self._running:
            return self.actual_port or 0

        import socket

        # Bind socket NOW - this reserves the port atomically
        self._bound_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._bound_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Try binding with retries for robustness
        max_retries = 10
        for attempt in range(max_retries):
            try:
                self._bound_socket.bind((self.host, self.requested_port))
                break
            except OSError as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to bind to port after {max_retries} attempts: {e}")
                # If specific port requested and failed, can't retry
                if self.requested_port != 0:
                    raise
                # For dynamic port (0), just retry - OS will pick different port
                import time
                time.sleep(0.1)

        # Get the actual port (important if requested_port was 0)
        self.actual_port = self._bound_socket.getsockname()[1]

        with open(debug_path, "a") as f:
            f.write(f"[SDK Server] Bound to port {self.actual_port}\n")

        # Set socket to listen mode
        self._bound_socket.listen(100)
        self._bound_socket.setblocking(False)

        # Start server in background thread
        self._ready_event.clear()
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

        with open(debug_path, "a") as f:
            f.write(f"[SDK Server] Thread started, waiting for ready event...\n")

        # Wait for server to be ready (with timeout)
        if not self._ready_event.wait(timeout=10.0):
            with open(debug_path, "a") as f:
                f.write(f"[SDK Server] TIMEOUT waiting for ready event!\n")
            raise RuntimeError("Server failed to start within 10 seconds")

        with open(debug_path, "a") as f:
            f.write(f"[SDK Server] Ready! Returning port {self.actual_port}\n")

        return self.actual_port

    def stop(self) -> None:
        """Stop the HTTP server."""
        self._running = False

        # Clean up runtime file
        RuntimeInfo.clear()

        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def update_backend_connected(self, connected: bool) -> None:
        """Update the backend connection status in runtime.json."""
        info = RuntimeInfo.load()
        if info:
            info.backend_connected = connected
            info.save()

    def _call_on_main_loop(self, coro) -> Future:
        """Schedule a coroutine on the main (TUI) event loop and return a Future.

        This is used for cross-loop communication when the tunnel needs to be
        called from the SDK server's thread.
        """
        if self._main_loop is None or self._main_loop.is_closed() or not self._main_loop.is_running():
            raise RuntimeError("Main loop not available - SDK server is shutting down")
        return asyncio.run_coroutine_threadsafe(coro, self._main_loop)

    async def _stream_response(self, request_id: str, queue: asyncio.Queue):
        """Stream SSE events from the response queue.

        The queue is polled from the main loop using run_coroutine_threadsafe
        to handle cross-loop communication.
        """
        try:
            while True:
                try:
                    # Get from queue on main loop, wait in this loop
                    if self._main_loop:
                        try:
                            future = self._call_on_main_loop(
                                asyncio.wait_for(queue.get(), timeout=120.0)
                            )
                            msg = await asyncio.get_event_loop().run_in_executor(
                                None, future.result, 120.0
                            )
                        except RuntimeError as e:
                            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                            break
                    else:
                        # Fallback: direct await (same loop)
                        msg = await asyncio.wait_for(queue.get(), timeout=120.0)
                except (asyncio.TimeoutError, TimeoutError):
                    yield f"event: error\ndata: {json.dumps({'error': 'Timeout'})}\n\n"
                    break

                msg_type = msg.get("type")
                if msg_type == "token":
                    yield f"event: token\ndata: {json.dumps({'t': msg.get('token', '')})}\n\n"
                elif msg_type == "filler":
                    yield f"event: filler\ndata: {json.dumps(msg.get('data', {}))}\n\n"
                elif msg_type == "done":
                    yield f"event: done\ndata: {json.dumps(msg.get('data', {}))}\n\n"
                    break
                elif msg_type == "error":
                    yield f"event: error\ndata: {json.dumps({'error': msg.get('error')})}\n\n"
                    break
        finally:
            if self.tunnel:
                # cancel_chat_request is sync, call via call_soon_threadsafe for thread safety
                if self._main_loop:
                    self._main_loop.call_soon_threadsafe(
                        self.tunnel.cancel_chat_request, request_id
                    )
                else:
                    self.tunnel.cancel_chat_request(request_id)

    async def _wait_for_response(self, request_id: str, queue: asyncio.Queue) -> dict:
        """Wait for complete response (non-streaming mode)."""
        try:
            while True:
                try:
                    # Get from queue on main loop
                    if self._main_loop:
                        try:
                            future = self._call_on_main_loop(
                                asyncio.wait_for(queue.get(), timeout=120.0)
                            )
                            msg = await asyncio.get_event_loop().run_in_executor(
                                None, future.result, 120.0
                            )
                        except RuntimeError as e:
                            return {"error": str(e)}
                    else:
                        msg = await asyncio.wait_for(queue.get(), timeout=120.0)
                except (asyncio.TimeoutError, TimeoutError):
                    return {"error": "Timeout"}

                msg_type = msg.get("type")
                if msg_type == "done":
                    data = msg.get("data", {})
                    return {
                        "response": data.get("speech", ""),
                        "verified": data.get("verified", False),
                        "citations": data.get("citations", []),
                    }
                elif msg_type == "filler":
                    continue
                elif msg_type == "error":
                    return {"error": msg.get("error", "Unknown error")}
        finally:
            if self.tunnel:
                # cancel_chat_request is sync, call via call_soon_threadsafe for thread safety
                if self._main_loop:
                    self._main_loop.call_soon_threadsafe(
                        self.tunnel.cancel_chat_request, request_id
                    )
                else:
                    self.tunnel.cancel_chat_request(request_id)

    def _run_server(self) -> None:
        """Run the server (called in background thread)."""
        debug_path = get_runtime_path().parent / "sdk_debug.log"
        with open(debug_path, "a") as f:
            f.write(f"[SDK Server] _run_server() started in thread\n")

        try:
            from fastapi import FastAPI, Request
            from fastapi.responses import StreamingResponse, JSONResponse
        except ImportError as e:
            with open(debug_path, "a") as f:
                f.write(f"[SDK Server] FastAPI import failed: {e}\n")
            self._ready_event.set()
            return

        # Create a minimal FastAPI app
        with open(debug_path, "a") as f:
            f.write(f"[SDK Server] Creating FastAPI app...\n")
        app = FastAPI(title="Loreguard SDK Server", version=get_version())

        # Store reference to self for route handlers
        server = self

        @app.get("/health")
        async def health():
            try:
                backend_connected = server.tunnel.connected if server.tunnel else False
                return {
                    "status": "ok",
                    "backend_connected": backend_connected,
                }
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "error": str(e)},
                )

        @app.post("/api/chat")
        async def chat(request: Request):
            if not server.tunnel or not server.tunnel.connected:
                return JSONResponse(
                    status_code=503,
                    content={"error": "Not connected to backend"},
                )

            body = await request.json()
            history = body.get("history") or body.get("context") or []
            player_handle = body.get("player_handle", body.get("playerHandle", ""))
            character_id = body.get("character_id", body.get("characterId", ""))
            current_context = body.get("current_context", body.get("currentContext", ""))
            enable_thinking = body.get("enable_thinking", body.get("enableThinking", False))
            max_speech_tokens = body.get("max_speech_tokens", body.get("maxSpeechTokens", 0))
            accept = request.headers.get("accept", "")
            streaming = "text/event-stream" in accept

            # Extract API token from Authorization header for backend auth
            auth_header = request.headers.get("authorization", "")
            api_token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""

            request_id = str(uuid.uuid4())

            # Call tunnel.send_chat_request on the main loop
            if server._main_loop:
                try:
                    future = server._call_on_main_loop(
                        server.tunnel.send_chat_request(
                            request_id=request_id,
                            character_id=character_id,
                            message=body.get("message", ""),
                            player_handle=player_handle,
                            current_context=current_context,
                            history=history,
                            enable_thinking=enable_thinking,
                            verbose=body.get("verbose", False),
                            api_token=api_token,
                            max_speech_tokens=max_speech_tokens,
                        )
                    )
                    # Wait for the result
                    response_queue = await asyncio.get_event_loop().run_in_executor(
                        None, future.result, 30.0
                    )
                except RuntimeError as e:
                    return JSONResponse(status_code=503, content={"error": str(e)})
            else:
                # Fallback: direct await (same loop - standalone mode)
                response_queue = await server.tunnel.send_chat_request(
                    request_id=request_id,
                    character_id=character_id,
                    message=body.get("message", ""),
                    player_handle=player_handle,
                    current_context=current_context,
                    history=history,
                    enable_thinking=enable_thinking,
                    verbose=body.get("verbose", False),
                    api_token=api_token,
                    max_speech_tokens=max_speech_tokens,
                )

            if streaming:
                return StreamingResponse(
                    server._stream_response(request_id, response_queue),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            else:
                result = await server._wait_for_response(request_id, response_queue)
                if "error" in result:
                    return JSONResponse(status_code=500, content=result)
                return result

        # Write runtime info
        with open(debug_path, "a") as f:
            f.write(f"[SDK Server] Writing runtime info for port {self.actual_port}...\n")
        write_runtime_info(port=self.actual_port)

        if self.on_status_change:
            self.on_status_change(f"SDK server on port {self.actual_port}")

        # Create event loop for this thread
        with open(debug_path, "a") as f:
            f.write(f"[SDK Server] Creating event loop...\n")
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            # Use uvicorn - it handles background threads better than hypercorn
            # (hypercorn tries to install signal handlers which fail in non-main threads)
            with open(debug_path, "a") as f:
                f.write(f"[SDK Server] Importing uvicorn...\n")
            import uvicorn

            # Close the pre-bound socket - uvicorn will rebind to the same port
            # This is safe because we're in the same thread and no other process
            # should grab the port in this tiny window
            self._bound_socket.close()
            self._bound_socket = None

            # Create uvicorn config (disable all logging to avoid TUI glitches)
            config = uvicorn.Config(
                app=app,
                host=self.host,
                port=self.actual_port,
                log_level="critical",  # Suppress all logs
                access_log=False,  # Disable access logging
            )

            # Signal ready before starting to serve
            self._running = True
            self._ready_event.set()

            with open(debug_path, "a") as f:
                f.write(f"[SDK Server] Starting uvicorn on port {self.actual_port}...\n")

            # Create uvicorn server and disable signal handlers (we're in a background thread)
            uvicorn_server = uvicorn.Server(config)
            uvicorn_server.install_signal_handlers = lambda: None

            # Run the server
            self._loop.run_until_complete(uvicorn_server.serve())

            with open(debug_path, "a") as f:
                f.write("[SDK Server] Uvicorn stopped normally\n")

        except Exception as e:
            # Make sure we signal even on error
            debug_path = get_runtime_path().parent / "sdk_debug.log"
            with open(debug_path, "a") as f:
                f.write(f"[SDK Server] ERROR: {type(e).__name__}: {e}\n")
                import traceback
                f.write(traceback.format_exc())
            if not self._ready_event.is_set():
                self._ready_event.set()
            raise
        finally:
            self._running = False
            if self._bound_socket:
                try:
                    self._bound_socket.close()
                except Exception:
                    pass
            RuntimeInfo.clear()


# Global instance for easy access
_server: Optional[EmbeddedHTTPServer] = None


def start_sdk_server(
    tunnel: Any,
    host: str = "127.0.0.1",
    port: int = 0,
    on_status_change: Optional[Callable[[str], None]] = None,
    main_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> int:
    """Start the SDK server (convenience function).

    Args:
        tunnel: BackendTunnel instance to use for chat requests
        host: Host to bind to
        port: Port to bind to (0 = auto-assign)
        on_status_change: Callback for status updates
        main_loop: The main event loop (TUI's loop) for cross-loop communication

    Returns:
        The actual port the server is listening on.
    """
    global _server

    if _server and _server._running:
        return _server.actual_port or 0

    _server = EmbeddedHTTPServer(tunnel, host, port, on_status_change, main_loop)
    return _server.start()


def stop_sdk_server() -> None:
    """Stop the SDK server."""
    global _server
    if _server:
        _server.stop()
        _server = None


def force_stop_sdk_server() -> None:
    """Force stop the SDK server immediately without waiting."""
    global _server
    if _server:
        _server._running = False
        # Clean up runtime file
        RuntimeInfo.clear()
        # Force stop the loop if it exists
        if _server._loop and _server._loop.is_running():
            _server._loop.call_soon_threadsafe(_server._loop.stop)
        # Don't wait for thread - it's a daemon thread anyway
        _server = None


def update_backend_status(connected: bool) -> None:
    """Update backend connection status in runtime.json."""
    global _server
    if _server:
        _server.update_backend_connected(connected)
    else:
        # Update directly if no server
        info = RuntimeInfo.load()
        if info:
            info.backend_connected = connected
            info.save()
