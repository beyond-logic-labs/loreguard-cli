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
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from rich.console import Console

from .tunnel import BackendTunnel
from .llm import LLMProxy
from .config import get_config_value
from .nli import NLIService, is_nli_model_available

load_dotenv()

console = Console()

# =============================================================================
# App Setup
# =============================================================================

app = FastAPI(
    title="Loreguard Client",
    description="Bridge between local LLM and Loreguard backend",
    version="0.3.0",
)

# Global instances
tunnel: BackendTunnel | None = None
llm_proxy: LLMProxy | None = None
nli_service: NLIService | None = None


@app.on_event("startup")
async def startup():
    """Initialize connections on startup."""
    global tunnel, llm_proxy, nli_service

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
# CLI Entry Point
# =============================================================================

def run():
    """Run the client bridge."""
    console.print("[bold green]Loreguard Client[/bold green]")
    console.print("=" * 40)

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8081"))

    console.print(f"Starting server at [cyan]http://{host}:{port}[/cyan]")
    console.print("Press Ctrl+C to stop\n")

    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=os.getenv("DEV", "false").lower() == "true",
    )


if __name__ == "__main__":
    run()
