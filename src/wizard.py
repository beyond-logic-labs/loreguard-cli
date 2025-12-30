#!/usr/bin/env python3
"""Loreguard Wizard - Interactive terminal setup wizard.

Arrow-key navigation, colorful UI, works on any terminal.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from .term_ui import (
    Menu,
    MenuItem,
    InputField,
    ProgressDisplay,
    StatusDisplay,
    Colors,
    supports_color,
    show_cursor,
    print_header,
    print_success,
    print_error,
    print_info,
)

# Quiet logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("loreguard")


def _c(color: str) -> str:
    """Return color code if supported."""
    return color if supports_color() else ""


def print_banner():
    """Print the startup banner."""
    c = _c
    banner = f"""
{c(Colors.CYAN)}┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  {c(Colors.BRIGHT_CYAN)}██╗      ██████╗ ██████╗ ███████╗ {c(Colors.BRIGHT_MAGENTA)}██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗ {c(Colors.CYAN)} │
│  {c(Colors.BRIGHT_CYAN)}██║     ██╔═══██╗██╔══██╗██╔════╝{c(Colors.BRIGHT_MAGENTA)}██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗{c(Colors.CYAN)} │
│  {c(Colors.BRIGHT_CYAN)}██║     ██║   ██║██████╔╝█████╗  {c(Colors.BRIGHT_MAGENTA)}██║  ███╗██║   ██║███████║██████╔╝██║  ██║{c(Colors.CYAN)} │
│  {c(Colors.BRIGHT_CYAN)}██║     ██║   ██║██╔══██╗██╔══╝  {c(Colors.BRIGHT_MAGENTA)}██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║{c(Colors.CYAN)} │
│  {c(Colors.BRIGHT_CYAN)}███████╗╚██████╔╝██║  ██║███████╗{c(Colors.BRIGHT_MAGENTA)}╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝{c(Colors.CYAN)} │
│  {c(Colors.BRIGHT_CYAN)}╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝ {c(Colors.BRIGHT_MAGENTA)}╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ {c(Colors.CYAN)} │
│                                                                              │
│  {c(Colors.WHITE)}Local inference for your game NPCs{c(Colors.CYAN)}                                       │
│  {c(Colors.GRAY)}loreguard.com{c(Colors.CYAN)}                                                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘{c(Colors.RESET)}
"""
    print(banner)


async def step_authentication() -> tuple[Optional[str], Optional[str], bool]:
    """Step 1: Get and validate token.

    Returns: (token, worker_id, dev_mode) or (None, None, False) if cancelled.
    """
    print_info("Step 1/3: Authentication")
    print()

    # First, ask how they want to authenticate
    auth_menu = Menu(
        items=[
            MenuItem(
                label="Login with browser",
                value="browser",
                description="Opens loreguard.com to authenticate",
            ),
            MenuItem(
                label="Paste token",
                value="token",
                description="Manually enter your worker token",
            ),
            MenuItem(
                label="Dev mode",
                value="dev",
                description="Test locally without backend connection",
            ),
        ],
        title="Authentication",
        prompt="How do you want to authenticate?",
    )

    auth_choice = auth_menu.run()

    if auth_choice is None:
        return None, None, False

    # Dev mode
    if auth_choice.value == "dev":
        print_success("Dev mode enabled (no backend connection)")
        print()
        return "dev_mock_token", "dev-worker", True

    # Browser authentication
    if auth_choice.value == "browser":
        return await _auth_with_browser()

    # Manual token entry
    return await _auth_with_token()


async def _auth_with_browser() -> tuple[Optional[str], Optional[str], bool]:
    """Authenticate via browser OAuth flow."""
    import httpx
    import webbrowser
    import secrets

    status = StatusDisplay(title="Browser Authentication", height=6)
    status.set_line(0, "Status", "Starting authentication...")
    status.draw()

    # Generate a device code / session ID
    session_id = secrets.token_urlsafe(16)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Request device auth
            response = await client.post(
                "https://api.loreguard.com/api/workers/device-auth",
                json={"session_id": session_id},
            )

            if response.status_code == 200:
                data = response.json()
                auth_url = data.get("auth_url", f"https://loreguard.com/cli-auth?session={session_id}")

                status.set_line(0, "Status", "Opening browser...")
                status.set_line(1, "URL", auth_url[:50] + "..." if len(auth_url) > 50 else auth_url)
                status.draw()

                # Open browser
                webbrowser.open(auth_url)

                status.set_line(0, "Status", "Waiting for authorization...")
                status.set_line(2, "", "Complete login in your browser")
                status.draw()

                # Poll for completion
                for _ in range(120):  # 2 minute timeout
                    await asyncio.sleep(2)
                    poll_response = await client.get(
                        f"https://api.loreguard.com/api/workers/device-auth/{session_id}"
                    )
                    if poll_response.status_code == 200:
                        poll_data = poll_response.json()
                        if poll_data.get("status") == "completed":
                            token = poll_data.get("token")
                            worker_id = poll_data.get("worker_id", "worker")
                            status.clear()
                            print_success(f"Authenticated as {worker_id}")
                            print()
                            return token, worker_id, False

                status.clear()
                print_error("Authentication timed out")
                print()
                return await step_authentication()

            elif response.status_code == 404:
                # Endpoint not ready - fall back to manual token
                status.clear()
                print_info("Browser auth not available yet, please paste token")
                print()
                return await _auth_with_token()
            else:
                status.clear()
                print_error("Failed to start browser authentication")
                print()
                return await step_authentication()

    except httpx.ConnectError:
        status.clear()
        print_error("Cannot connect to server")
        print()
        return await step_authentication()

    except Exception as e:
        status.clear()
        print_error(f"Error: {e}")
        print()
        return await step_authentication()


async def _auth_with_token() -> tuple[Optional[str], Optional[str], bool]:
    """Authenticate with manually entered token."""
    import httpx

    def validate_token(value: str) -> Optional[str]:
        if not value:
            return "Token is required"
        if not value.startswith("lg_worker_"):
            return "Token must start with 'lg_worker_'"
        return None

    input_field = InputField(
        prompt="Enter your worker token:",
        password=True,
        validator=validate_token,
    )

    token = input_field.run(title="Paste Token")

    if token is None:
        return await step_authentication()

    # Validate with server
    status = StatusDisplay(title="Validating Token", height=5)
    status.set_line(0, "Status", "Connecting to server...")
    status.draw()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "https://api.loreguard.com/api/workers/validate",
                json={"token": token},
            )
            status.clear()

            if response.status_code == 200:
                data = response.json()
                worker_id = data.get("workerId", "worker")
                print_success(f"Authenticated as {worker_id}")
                print()
                return token, worker_id, False
            elif response.status_code == 404:
                print_success("Token accepted")
                print()
                return token, "worker", False
            else:
                print_error("Invalid token")
                print()
                return await _auth_with_token()

    except httpx.ConnectError:
        status.clear()
        print_error("Cannot connect to server")
        print()
        return await _auth_with_token()

    except Exception as e:
        status.clear()
        print_error(f"Error: {e}")
        print()
        return await _auth_with_token()


async def step_model_selection() -> Optional[Path]:
    """Step 2: Select and optionally download a model."""
    print_info("Step 2/3: Model Selection")
    print()

    from .models_registry import SUPPORTED_MODELS
    from .llama_server import get_models_dir

    models_dir = get_models_dir()

    # Check which models are installed
    installed_ids = set()
    for model in SUPPORTED_MODELS:
        if (models_dir / model.filename).exists():
            installed_ids.add(model.id)

    # Build menu items
    items = []
    for model in SUPPORTED_MODELS:
        if model.id in installed_ids:
            tag = "✓ installed"
        else:
            tag = f"{model.size_gb:.1f} GB"

        if model.recommended:
            tag += " • recommended"
        if model.experimental:
            tag += " • experimental"

        # Show hardware requirements in description
        desc = f"{model.hardware}"

        items.append(MenuItem(
            label=model.name,
            value=model.id,
            description=desc,
            tag=tag,
        ))

    items.append(MenuItem(
        label="Custom model path...",
        value="__custom__",
        description="Enter path to your own .gguf file",
        tag="",
    ))

    menu = Menu(
        items=items,
        title="Select Model",
        prompt="Choose a model to use:",
    )

    selected = menu.run()

    if selected is None:
        return None

    if selected.value == "__custom__":
        input_field = InputField(prompt="Enter path to .gguf file:")
        custom_path = input_field.run(title="Custom Model")

        if custom_path is None:
            return await step_model_selection()

        if custom_path.startswith("~"):
            custom_path = str(Path.home()) + custom_path[1:]

        model_path = Path(custom_path)

        if model_path.is_dir():
            gguf_files = list(model_path.glob("**/*.gguf"))
            if gguf_files:
                model_path = gguf_files[0]
                print_success(f"Found: {model_path.name}")
            else:
                print_error("No .gguf files found in directory")
                return await step_model_selection()

        if not model_path.exists():
            print_error(f"File not found: {model_path}")
            return await step_model_selection()

        print_success(f"Using: {model_path.name}")
        print()
        return model_path

    # Find selected model
    model = None
    for m in SUPPORTED_MODELS:
        if m.id == selected.value:
            model = m
            break

    if model is None:
        return None

    model_path = models_dir / model.filename

    if model_path.exists():
        print_success(f"Model ready: {model.name}")
        print()
    else:
        model_path = await download_model(model, model_path)
        if model_path is None:
            return await step_model_selection()
        print_success(f"Downloaded: {model.name}")
        print()

    return model_path


async def download_model(model, dest: Path) -> Optional[Path]:
    """Download a model with progress display."""
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)

    progress = ProgressDisplay(
        title=f"Downloading {model.name}",
        total=model.size_bytes or 1,
        subtitle=model.url,
    )

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
            async with client.stream("GET", model.url) as response:
                response.raise_for_status()
                total = model.size_bytes or int(response.headers.get("content-length", 0))
                progress.total = total
                downloaded = 0

                with open(dest, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress.update(
                            downloaded,
                            f"{downloaded // 1024 // 1024} MB / {total // 1024 // 1024} MB"
                        )

        progress.clear()
        return dest

    except Exception as e:
        progress.clear()
        print_error(f"Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return None


async def step_start(
    model_path: Path,
    token: str,
    worker_id: str,
    dev_mode: bool,
) -> int:
    """Step 3: Start llama-server and connect to backend."""
    print_info("Step 3/3: Starting Services")
    print()

    from .llama_server import (
        LlamaServerProcess,
        is_llama_server_installed,
        download_llama_server,
        DownloadProgress,
    )

    status = StatusDisplay(title="Loreguard", height=12)

    # Download llama-server if needed
    if not is_llama_server_installed():
        status.set_line(0, "llama-server", "Downloading...")
        status.draw()

        try:
            def on_progress(msg: str, prog: DownloadProgress | None):
                if prog:
                    status.set_line(0, "llama-server", f"Downloading... {int(prog.percent)}%")
                    status.draw()

            await download_llama_server(on_progress)
            status.set_line(0, "llama-server", "✓ Downloaded")
            status.draw()
        except Exception as e:
            status.clear()
            print_error(f"Failed to download llama-server: {e}")
            return 1

    # Start llama-server
    status.set_line(0, "llama-server", "Starting...")
    status.set_line(1, "Model", model_path.name)
    status.draw()

    llama = LlamaServerProcess(model_path, port=8080)
    llama.start()

    status.set_line(0, "llama-server", "Loading model...")
    status.draw()

    ready = await llama.wait_for_ready(timeout=120.0)
    if not ready:
        status.clear()
        print_error("llama-server failed to start (timeout)")
        llama.stop()
        return 1

    status.set_line(0, "llama-server", "✓ Running on port 8080")
    status.draw()

    # Connect to backend (unless dev mode)
    tunnel = None
    if not dev_mode:
        status.set_line(2, "Backend", "Connecting...")
        status.draw()

        try:
            from .tunnel import BackendTunnel
            from .llm import LLMProxy

            llm_proxy = LLMProxy("http://127.0.0.1:8080")

            tunnel = BackendTunnel(
                backend_url="wss://api.loreguard.com/workers",
                llm_proxy=llm_proxy,
                worker_id=worker_id,
                worker_token=token,
                model_id=model_path.stem,
            )

            asyncio.create_task(tunnel.connect())
            await asyncio.sleep(2)

            status.set_line(2, "Backend", "✓ Connected")
        except Exception as e:
            status.set_line(2, "Backend", f"✗ Failed: {e}")
            status.set_line(3, "", "  (local-only mode)")
    else:
        status.set_line(2, "Mode", "Dev (local only)")
        status.set_line(3, "API", "http://localhost:8080")

    status.clear()

    # Offer chat option (requires authentication with Loreguard API)
    if not dev_mode:
        mode_menu = Menu(
            items=[
                MenuItem(
                    label="Chat with NPC",
                    value="chat",
                    description="Interactive chat using Loreguard API",
                ),
                MenuItem(
                    label="Run as worker",
                    value="server",
                    description="Wait for inference requests from Loreguard",
                ),
            ],
            title="What would you like to do?",
            prompt="llama-server is ready. Your worker is connected to Loreguard.",
        )

        mode_choice = mode_menu.run()

        if mode_choice and mode_choice.value == "chat":
            from .npc_chat import run_npc_chat

            try:
                await run_npc_chat(api_token=token)
            except KeyboardInterrupt:
                pass

            # After chat, cleanup
            if tunnel:
                try:
                    await tunnel.disconnect()
                except:
                    pass
            llama.stop()
            print_success("Goodbye!")
            return 0
    else:
        # Dev mode - no API access, just run server
        print_info("Dev mode: llama-server running at http://localhost:8080")
        print_info("Chat with NPCs requires authentication (not dev mode).")
        print()

    # Running state (server mode)
    status = StatusDisplay(title="Loreguard Running", height=12)
    status.set_line(0, "llama-server", "✓ Running on port 8080")
    status.set_line(1, "Model", model_path.name)
    if dev_mode:
        status.set_line(2, "Mode", "Dev (local only)")
        status.set_line(3, "API", "http://localhost:8080")
    elif tunnel:
        status.set_line(2, "Backend", "✓ Connected")
    status.set_line(4, "", "")
    status.set_line(5, "Requests", "0")
    status.set_line(6, "Tokens", "0")
    status.set_footer("Ctrl+C to stop")
    status.draw()

    # Metrics tracking
    request_count = [0]
    total_tokens = [0]

    def on_request(npc: str, tokens: int, ttft_ms: float, total_ms: float):
        request_count[0] += 1
        total_tokens[0] += tokens
        tps = (tokens / total_ms * 1000) if total_ms > 0 else 0
        status.set_line(5, "Requests", str(request_count[0]))
        status.set_line(6, "Tokens", f"{total_tokens[0]:,}")
        status.set_line(7, "Last", f"{npc} • {tokens} tok • {tps:.1f} tk/s")
        status.draw()

    if tunnel:
        tunnel.on_request_complete = on_request

    # Wait for shutdown
    running = True

    def handle_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass

    # Cleanup
    status.set_title("Shutting Down")
    status.set_line(0, "llama-server", "Stopping...")
    status.set_footer("")
    status.draw()

    llama.stop()

    if tunnel:
        try:
            await tunnel.disconnect()
        except:
            pass

    status.clear()
    print_success("Goodbye!")
    print()
    return 0


async def run_wizard() -> int:
    """Run the setup wizard."""
    try:
        print_banner()

        # Step 1: Authentication
        token, worker_id, dev_mode = await step_authentication()
        if token is None:
            print_error("Cancelled")
            return 1

        # Step 2: Model Selection
        model_path = await step_model_selection()
        if model_path is None:
            print_error("Cancelled")
            return 1

        # Step 3: Start
        return await step_start(model_path, token, worker_id, dev_mode)

    except KeyboardInterrupt:
        print()
        print_error("Interrupted")
        return 1
    finally:
        show_cursor()


def main():
    """Entry point."""
    try:
        exit_code = asyncio.run(run_wizard())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        show_cursor()
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
