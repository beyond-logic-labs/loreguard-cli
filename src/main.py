"""Loreguard Client Bridge.

Bridge between llama.cpp server and Loreguard backend.

The client:
1. Connects to llama.cpp server (localhost:8080)
2. Connects via WebSocket to remote backend
3. Receives inference requests from backend
4. Executes on local llama.cpp with full sampling config
5. Returns results to backend (content, thinking, usage)

Features (from netshell's local_llm.go):
- Full sampling configuration (top_p, min_p, repeat_penalty, etc.)
- Stop sequences to prevent hallucinated conversation turns
- Thinking mode control (enable_thinking for Qwen3)
- JSON schema/response_format for structured output
- Thinking tag extraction (<think>...</think>)
"""

import asyncio
import json
import os
import uuid
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from rich.console import Console

from .tunnel import BackendTunnel
from .llm import LLMProxy
from .config import get_config_value
from .nli import NLIService, is_nli_model_available
from .intent_classifier import IntentClassifier, is_intent_model_available

load_dotenv()

console = Console()

# =============================================================================
# App Setup
# =============================================================================

from .runtime import get_version

app = FastAPI(
    title="Loreguard Client",
    description="Bridge between local LLM and Loreguard backend",
    version=get_version(),
)

# Global instances
tunnel: BackendTunnel | None = None
llm_proxy: LLMProxy | None = None
nli_service: NLIService | None = None
intent_classifier: IntentClassifier | None = None


@app.on_event("startup")
async def startup():
    """Initialize connections on startup."""
    global tunnel, llm_proxy, nli_service, intent_classifier

    # Note: runtime.json is written by run() before hypercorn starts
    # (socket is already bound at that point)

    # Initialize local LLM connection
    llm_url = get_config_value("LLM_ENDPOINT", "http://localhost:8080")
    console.print(f"[green]LLM endpoint:[/green] {llm_url}")
    llm_proxy = LLMProxy(llm_url)

    # Check LLM availability
    if await llm_proxy.check():
        console.print("[green]LLM is available[/green]")
        models = await llm_proxy.list_models()
        if models:
            console.print(f"[cyan]Available models:[/cyan] {', '.join(models[:5])}")
    else:
        console.print("[yellow]Warning: LLM not available yet[/yellow]")

    # Initialize NLI service (optional - for fact verification)
    enable_nli = os.getenv("LOREGUARD_NLI_ENABLED", "true").lower() == "true"
    if enable_nli:
        console.print("[cyan]Initializing NLI service...[/cyan]")
        nli_service = NLIService()
        if nli_service.load_model():
            console.print(f"[green]NLI service ready (device: {nli_service.device})[/green]")
        else:
            console.print("[yellow]Warning: NLI model failed to load[/yellow]")
            console.print("[yellow]  NLI capability will be disabled[/yellow]")
            nli_service = None
    else:
        console.print("[yellow]NLI service disabled (set LOREGUARD_NLI_ENABLED=true to enable)[/yellow]")

    # Initialize intent classifier (ADR-0010 - optional, for adaptive retrieval)
    enable_intent = os.getenv("LOREGUARD_INTENT_ENABLED", "true").lower() == "true"
    if enable_intent:
        console.print("[cyan]Initializing intent classifier...[/cyan]")
        intent_classifier = IntentClassifier()
        if intent_classifier.load_model():
            console.print(f"[green]Intent classifier ready (device: {intent_classifier.device})[/green]")
        else:
            console.print("[yellow]Warning: Intent model failed to load[/yellow]")
            console.print("[yellow]  Intent classification will be disabled[/yellow]")
            intent_classifier = None
    else:
        console.print("[yellow]Intent classifier disabled (set LOREGUARD_INTENT_ENABLED=true to enable)[/yellow]")

    # Connect to remote backend
    backend_url = get_config_value("BACKEND_URL", "wss://api.lorekeeper.ai/workers")
    worker_id = get_config_value("WORKER_ID", "")
    worker_token = get_config_value("WORKER_TOKEN", "")
    model_id = get_config_value("MODEL_ID", "default")

    if backend_url and worker_id and worker_token:
        console.print(f"[green]Connecting to backend:[/green] {backend_url}")
        console.print(f"[green]Worker ID:[/green] {worker_id}")
        tunnel = BackendTunnel(
            backend_url, llm_proxy, worker_id, worker_token, model_id,
            nli_service=nli_service,
            intent_classifier=intent_classifier,
        )
        asyncio.create_task(tunnel.connect())
    elif backend_url:
        console.print("[yellow]Warning: WORKER_ID and WORKER_TOKEN required for backend connection[/yellow]")
        console.print("[yellow]Generate a token using:[/yellow]")
        console.print("  go run cmd/token/main.go generate -local -worker-id <id> -model-id <model>")
    else:
        console.print("[yellow]Warning: No backend URL configured[/yellow]")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    from .runtime import RuntimeInfo

    # Cleanup runtime file
    RuntimeInfo.clear()
    console.print("[dim]Runtime info cleared[/dim]")

    if llm_proxy:
        await llm_proxy.close()
    if tunnel:
        await tunnel.disconnect()


# =============================================================================
# Health Check Endpoint
# =============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    llm_available = await llm_proxy.check() if llm_proxy else False
    return {
        "status": "ok",
        "llm_available": llm_available,
        "backend_connected": tunnel.connected if tunnel else False,
        "nli_available": nli_service.is_loaded if nli_service else False,
    }


@app.get("/models")
async def list_models():
    """List available LLM models."""
    if not llm_proxy:
        return {"models": [], "error": "LLM proxy not initialized"}

    models = await llm_proxy.list_models()
    return {"models": models}


# =============================================================================
# Chat Endpoint (Local Proxy Architecture)
# =============================================================================

class ChatRequest(BaseModel):
    """Request body for /api/chat endpoint."""
    character_id: str
    message: str
    player_handle: str = ""
    current_context: str = ""
    history: list[dict[str, Any]] = Field(default_factory=list)
    enable_thinking: bool = False
    verbose: bool = False


@app.post("/api/chat")
async def chat(request: Request, body: ChatRequest):
    """Chat endpoint for game clients.

    This is the local proxy endpoint that games use to talk to NPCs.
    The request is forwarded to the Loreguard backend via WebSocket,
    and the response is streamed back as SSE.

    Headers:
        Accept: text/event-stream  -> Stream tokens as SSE events
        Accept: application/json   -> Return full JSON response (wait for completion)

    SSE Event Format:
        event: token
        data: {"t": "Hello"}

        event: done
        data: {"speech": "Hello, traveler...", "verified": true, ...}

        event: error
        data: {"error": "Error message"}
    """
    if not tunnel or not tunnel.connected:
        return JSONResponse(
            status_code=503,
            content={"error": "Not connected to backend"},
        )

    # Check if streaming is requested
    accept = request.headers.get("accept", "")
    streaming = "text/event-stream" in accept

    # Generate request ID
    request_id = str(uuid.uuid4())

    # Send chat request and get response queue
        response_queue = await tunnel.send_chat_request(
            request_id=request_id,
            character_id=body.character_id,
            message=body.message,
            player_handle=body.player_handle,
            current_context=body.current_context,
            history=body.history,
            enable_thinking=body.enable_thinking,
            verbose=body.verbose,
        )

    if streaming:
        # Stream tokens as SSE
        return StreamingResponse(
            _stream_chat_response(request_id, response_queue),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )
    else:
        # Wait for complete response
        result = await _wait_for_chat_response(request_id, response_queue)
        if "error" in result:
            return JSONResponse(
                status_code=500,
                content=result,
            )
        return result


async def _stream_chat_response(request_id: str, queue: asyncio.Queue):
    """Generate SSE events from chat response queue."""
    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                yield f"event: error\ndata: {json.dumps({'error': 'Request timeout'})}\n\n"
                break

            msg_type = msg.get("type")

            if msg_type == "token":
                yield f"event: token\ndata: {json.dumps({'t': msg.get('token', '')})}\n\n"

            elif msg_type == "done":
                yield f"event: done\ndata: {json.dumps(msg.get('data', {}))}\n\n"
                break

            elif msg_type == "error":
                yield f"event: error\ndata: {json.dumps({'error': msg.get('error', 'Unknown error')})}\n\n"
                break

    except asyncio.CancelledError:
        # Client disconnected
        if tunnel:
            tunnel.cancel_chat_request(request_id)
        raise
    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
    finally:
        # Clean up pending request
        if tunnel:
            tunnel.cancel_chat_request(request_id)


async def _wait_for_chat_response(request_id: str, queue: asyncio.Queue) -> dict:
    """Wait for complete chat response (non-streaming mode)."""
    try:
        tokens = []
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                return {"error": "Request timeout"}

            msg_type = msg.get("type")

            if msg_type == "token":
                tokens.append(msg.get("token", ""))

            elif msg_type == "done":
                data = msg.get("data", {})
                return {
                    "response": data.get("speech", ""),
                    "character_id": "",  # Not available in response
                    "verified": data.get("verified", False),
                    "citations": data.get("citations", []),
                    "thoughts": data.get("thoughts", ""),
                    "metadata": {
                        "retries": data.get("retries", 0),
                        "latency_ms": data.get("latency_ms", 0),
                    },
                }

            elif msg_type == "error":
                return {"error": msg.get("error", "Unknown error")}

    except Exception as e:
        return {"error": str(e)}
    finally:
        if tunnel:
            tunnel.cancel_chat_request(request_id)


# =============================================================================
# CLI Entry Point
# =============================================================================

# Global to store port for runtime info (set by run(), read by startup event)
_server_port: int = 8081


def run():
    """Run the client bridge.

    Port is always dynamically assigned by the OS for security and to avoid conflicts.
    Use `loreguard status` or read runtime.json to discover the port.
    Set PORT env var to override with a specific port if needed.

    Uses socket-first binding for 100% race-condition-free port allocation:
    1. Bind socket immediately (reserves port atomically)
    2. Pass bound socket directly to hypercorn
    3. Never release the socket until server shutdown
    """
    global _server_port
    from .runtime import RuntimeInfo, get_runtime_path, write_runtime_info
    import socket
    import asyncio
    from hypercorn.asyncio import serve as hypercorn_serve
    from hypercorn.config import Config as HypercornConfig

    console.print("[bold green]Loreguard Client[/bold green]")
    console.print("=" * 40)

    host = os.getenv("HOST", "127.0.0.1")

    # Always use dynamic port unless explicitly overridden via PORT env var
    requested_port = int(os.getenv("PORT", "0"))

    # Socket-first binding - reserves port atomically with no race condition
    bound_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    bound_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        bound_socket.bind((host, requested_port))
    except OSError as e:
        console.print(f"[red]Failed to bind to port: {e}[/red]")
        bound_socket.close()
        return

    port = bound_socket.getsockname()[1]
    bound_socket.listen(100)
    bound_socket.setblocking(False)

    _server_port = port

    # Write runtime info for SDK discovery
    write_runtime_info(port=port)

    console.print(f"[green]Runtime info:[/green] {get_runtime_path()}")
    console.print(f"Starting server at [cyan]http://{host}:{port}[/cyan]")
    console.print("[dim]Use 'loreguard status' to get connection info[/dim]")
    console.print("Press Ctrl+C to stop\n")

    try:
        # Configure hypercorn to use our pre-bound socket
        hconfig = HypercornConfig()
        hconfig.bind = [f"fd://{bound_socket.fileno()}"]
        hconfig.accesslog = "-"  # Log to stdout
        hconfig.errorlog = "-"

        # Check for dev mode
        reload = os.getenv("DEV", "false").lower() == "true"
        if reload:
            console.print("[yellow]Warning: reload mode not supported with socket binding[/yellow]")

        # Run server with pre-bound socket
        asyncio.run(hypercorn_serve(app, hconfig))

    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        try:
            bound_socket.close()
        except Exception:
            pass
        RuntimeInfo.clear()


if __name__ == "__main__":
    run()
