#!/usr/bin/env python3
"""Loreguard Wizard - Interactive terminal setup wizard.

Uses Rich for display and InquirerPy for interactive prompts.
"""

import asyncio
import logging
import platform
import signal
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

from .term_ui import (
    ProgressDisplay,
    LiveStatusDisplay,
    console,
    print_header,
    print_success,
    print_error,
    print_info,
    show_cursor,
)

# Logger instance - configured by main()
log = logging.getLogger("loreguard")


def _configure_logging(verbose: bool = False) -> Optional[Path]:
    """Configure logging level based on verbose flag."""
    if verbose:
        log_file = Path.cwd() / "loreguard-debug.log"

        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.DEBUG)
        log.setLevel(logging.DEBUG)
        log.debug("Verbose mode enabled - logging to %s", log_file)
        return log_file
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")
        log.setLevel(logging.WARNING)
        return None


def print_banner():
    """Print the startup banner using Rich."""
    console.print()
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print()
    console.print("  [bold bright_cyan]LORE[/bold bright_cyan][bold bright_magenta]GUARD[/bold bright_magenta]")
    console.print("  [dim]Local inference for your game NPCs[/dim]")
    console.print("  [dim]loreguard.com[/dim]")
    console.print()
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print()


@dataclass
class HardwareInfo:
    cpu: str
    ram_gb: Optional[float]
    gpu: str
    gpu_vram_gb: Optional[float]
    gpu_mem_type: Optional[str]


def _run_cmd(args: list[str]) -> str:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _get_cpu_name() -> str:
    cpu = platform.processor().strip()
    if cpu:
        return cpu

    if sys.platform == "darwin":
        cpu = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            cpu = ""
    elif sys.platform.startswith("win"):
        cpu = _run_cmd(["wmic", "cpu", "get", "name"])
        lines = [line.strip() for line in cpu.splitlines() if line.strip() and "name" not in line.lower()]
        cpu = lines[0] if lines else ""

    return cpu or "Unknown"


def _get_ram_gb() -> Optional[float]:
    ram_bytes = None
    if sys.platform == "darwin":
        mem = _run_cmd(["sysctl", "-n", "hw.memsize"])
        if mem.isdigit():
            ram_bytes = int(mem)
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            ram_bytes = int(parts[1]) * 1024
                        break
        except Exception:
            ram_bytes = None
    elif sys.platform.startswith("win"):
        try:
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint),
                    ("dwMemoryLoad", ctypes.c_uint),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            ram_bytes = int(status.ullTotalPhys)
        except Exception:
            ram_bytes = None

    if not ram_bytes:
        return None
    return round(ram_bytes / (1024 ** 3), 1)


def _parse_vram_to_gb(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    parts = value.replace(",", "").split()
    if len(parts) < 2:
        return None
    try:
        amount = float(parts[0])
    except ValueError:
        return None
    unit = parts[1].lower()
    if unit.startswith("gb"):
        return round(amount, 1)
    if unit.startswith("mb"):
        return round(amount / 1024.0, 1)
    if unit.startswith("kb"):
        return round(amount / (1024.0 * 1024.0), 2)
    return None


def _get_gpu_info() -> tuple[str, Optional[float], Optional[str]]:
    if sys.platform == "darwin":
        output = _run_cmd(["system_profiler", "SPDisplaysDataType"])
        names: list[str] = []
        vram: Optional[float] = None
        mem_type: Optional[str] = None
        for line in output.splitlines():
            if "Chipset Model:" in line:
                names.append(line.split(":", 1)[1].strip())
            if "VRAM (Total):" in line:
                vram = _parse_vram_to_gb(line.split(":", 1)[1])
            if "VRAM (Dynamic, Max):" in line and vram is None:
                vram = _parse_vram_to_gb(line.split(":", 1)[1])
            if "Memory Type:" in line:
                mem_type = line.split(":", 1)[1].strip()
        if names:
            return ", ".join(names), vram, mem_type

    elif sys.platform.startswith("linux"):
        output = _run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,memory.type", "--format=csv,noheader"])
        if output:
            names = []
            vram = None
            mem_type = None
            for line in output.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if not parts:
                    continue
                names.append(parts[0])
                if len(parts) > 1 and vram is None:
                    vram = _parse_vram_to_gb(parts[1])
                if len(parts) > 2 and mem_type is None:
                    mem_type = parts[2] or None
            if names:
                return ", ".join(names), vram, mem_type

        output = _run_cmd(["lspci"])
        gpus = []
        for line in output.splitlines():
            lower = line.lower()
            if "vga compatible controller" in lower or "3d controller" in lower:
                gpus.append(line.split(":", 2)[-1].strip())
        if gpus:
            return ", ".join(gpus), None, None

    elif sys.platform.startswith("win"):
        output = _run_cmd(["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM"])
        names = []
        vram = None
        for line in output.splitlines():
            if not line.strip() or "name" in line.lower():
                continue
            parts = [part for part in line.split() if part.strip()]
            if not parts:
                continue
            if parts[-1].isdigit():
                ram_bytes = int(parts[-1])
                vram = round(ram_bytes / (1024 ** 3), 1) if vram is None else vram
                names.append(" ".join(parts[:-1]))
            else:
                names.append(" ".join(parts))
        if names:
            return ", ".join(names), vram, None

    return "Unknown", None, None


def detect_hardware() -> HardwareInfo:
    log.debug("Detecting hardware...")
    cpu = _get_cpu_name()
    log.debug(f"CPU: {cpu}")
    ram_gb = _get_ram_gb()
    log.debug(f"RAM: {ram_gb} GB")
    gpu_name, gpu_vram_gb, gpu_mem_type = _get_gpu_info()
    log.debug(f"GPU: {gpu_name}, VRAM: {gpu_vram_gb} GB, type: {gpu_mem_type}")
    return HardwareInfo(
        cpu=cpu,
        ram_gb=ram_gb,
        gpu=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        gpu_mem_type=gpu_mem_type,
    )


def _format_ram_gb(ram_gb: Optional[float]) -> str:
    return f"{ram_gb:.1f} GB" if ram_gb is not None else "Unknown"


def _format_gpu_info(hardware: HardwareInfo) -> str:
    parts = [hardware.gpu]
    if hardware.gpu_vram_gb is not None:
        parts.append(f"{hardware.gpu_vram_gb:.1f} GB VRAM")
    if hardware.gpu_mem_type:
        parts.append(hardware.gpu_mem_type)
    return " | ".join(parts)


def _is_shared_memory_gpu(hardware: HardwareInfo) -> bool:
    if not hardware.gpu_mem_type:
        return False
    mem_type = hardware.gpu_mem_type.lower()
    return "unified" in mem_type or "shared" in mem_type


def _effective_vram_gb(hardware: Optional[HardwareInfo]) -> Optional[float]:
    if not hardware:
        return None
    if _is_shared_memory_gpu(hardware):
        return None
    return hardware.gpu_vram_gb


def _classify_model_fit(model_size_gb: float, hardware: Optional[HardwareInfo]) -> str:
    if not hardware:
        return "unknown"
    vram_gb = _effective_vram_gb(hardware)
    usable_ram = max(0.0, hardware.ram_gb - 2.0) if hardware.ram_gb is not None else None

    if vram_gb is not None:
        if model_size_gb <= vram_gb:
            return "fits_vram"
        if usable_ram is not None and model_size_gb <= usable_ram:
            return "ram_spill"
        return "too_big"

    if usable_ram is not None:
        if model_size_gb <= usable_ram:
            return "fits_ram"
        return "too_big"

    return "unknown"


def _resolve_backend_model_id(filename_stem: str) -> str:
    """Map local model filename to backend-accepted model ID."""
    MODEL_MAPPINGS = {
        "qwen3-4b": "qwen3-4b",
        "qwen3-8b": "qwen3-8b",
        "qwen3-0.6b": "qwen3-0.6b",
        "qwen3-1.7b": "qwen3-4b",
        "llama-3": "llama-3.1-8b",
        "mistral": "mistral-7b",
        "phi-3": "phi-3",
        "tinyllama": "tinyllama",
    }

    search_str = filename_stem.lower()
    for pattern, backend_id in MODEL_MAPPINGS.items():
        if pattern in search_str:
            log.debug(f"Model ID mapped: {search_str} -> {backend_id}")
            return backend_id

    log.debug(f"Using 'external' model ID for: {search_str}")
    return "external"


def _suggest_model_id(models, hardware: Optional[HardwareInfo]) -> Optional[str]:
    if not hardware:
        return None

    ram = hardware.ram_gb
    if ram is None:
        ram = _effective_vram_gb(hardware)
    if ram is None:
        return None
    if ram >= 24:
        preferred = ["gpt-oss-20b", "qwen3-8b", "rnj-1-instruct", "qwen3-4b-instruct", "llama-3.2-3b-instruct", "qwen3-1.7b"]
    elif ram >= 16:
        preferred = ["qwen3-8b", "rnj-1-instruct", "qwen3-4b-instruct", "llama-3.2-3b-instruct", "qwen3-1.7b"]
    elif ram >= 12:
        preferred = ["qwen3-8b", "rnj-1-instruct", "qwen3-4b-instruct", "llama-3.2-3b-instruct", "qwen3-1.7b"]
    elif ram >= 8:
        preferred = ["qwen3-4b-instruct", "llama-3.2-3b-instruct", "qwen3-1.7b"]
    elif ram >= 6:
        preferred = ["llama-3.2-3b-instruct", "qwen3-1.7b"]
    else:
        preferred = ["qwen3-1.7b"]

    model_by_id = {model.id: model for model in models}
    ranked = {"fits_vram": [], "ram_spill": [], "fits_ram": []}

    for model_id in preferred:
        model = model_by_id.get(model_id)
        if not model:
            continue
        fit = _classify_model_fit(model.size_gb, hardware)
        if fit in ranked:
            ranked[fit].append(model_id)

    for bucket in ("fits_vram", "ram_spill", "fits_ram"):
        if ranked[bucket]:
            return ranked[bucket][0]
    return None


async def step_authentication() -> tuple[Optional[str], Optional[str], bool]:
    """Step 1: Get and validate token."""
    log.debug("Starting authentication step")
    print_info("Step 1/4: Authentication")
    print()

    auth_choice = inquirer.select(
        message="Choose an authentication method:",
        choices=[
            Choice(value="token", name="Paste token - Manually enter your API token"),
            Choice(value="dev", name="Dev mode - Test locally without backend connection"),
        ],
        default="token",
        pointer=">"
    ).execute()

    if auth_choice is None:
        return None, None, False

    if auth_choice == "dev":
        log.debug("User selected dev mode")
        print_success("Dev mode enabled (no backend connection)")
        print()
        return "dev_mock_token", "dev-worker", True

    log.debug("User selected token authentication")
    return await _auth_with_token()


async def _auth_with_token() -> tuple[Optional[str], Optional[str], bool]:
    """Authenticate with manually entered API token."""
    import httpx
    import socket

    token = inquirer.secret(
        message="Enter your API token:",
        validate=lambda x: len(x) > 0,
        invalid_message="Token is required",
    ).execute()

    if not token:
        return await step_authentication()

    log.debug("Validating token with API server...")
    console.print("[dim]Validating token...[/dim]")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            log.debug("Sending request to https://api.loreguard.com/api/auth/me")
            response = await client.get(
                "https://api.loreguard.com/api/auth/me",
                headers={"Authorization": f"Bearer {token}"},
            )
            log.debug(f"Auth response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                log.debug(f"Auth response: {data}")
                display_name = data.get("studio", {}).get("name") or data.get("email", "user")
                log.debug(f"Authenticated as: {display_name}")
                print_success(f"Authenticated as {display_name}")
                print()
                hostname = socket.gethostname() or "worker"
                worker_id = hostname.split(".")[0].replace(" ", "-")
                return token, worker_id, False
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", response.text[:100])
                except Exception:
                    error_msg = response.text[:100] if response.text else "Unknown error"

                print_error(f"Authentication failed: {error_msg}")
                print()
                return await _auth_with_token()

    except httpx.ConnectError as e:
        print_error(f"Cannot connect to server: {e}")
        print()
        return await _auth_with_token()

    except Exception as e:
        print_error(f"Error: {e}")
        print()
        return await _auth_with_token()


async def step_model_selection(hardware: Optional[HardwareInfo]) -> Optional[Path]:
    """Step 2: Select and optionally download a model."""
    log.debug("Starting model selection step")
    print_info("Step 2/4: Model Selection")
    print()

    from .models_registry import SUPPORTED_MODELS
    from .llama_server import get_models_dir

    models_dir = get_models_dir()
    log.debug(f"Models directory: {models_dir}")
    suggested_id = _suggest_model_id(SUPPORTED_MODELS, hardware)
    log.debug(f"Suggested model ID: {suggested_id}")

    installed_ids = set()
    for model in SUPPORTED_MODELS:
        if (models_dir / model.filename).exists():
            installed_ids.add(model.id)
    log.debug(f"Installed models: {installed_ids if installed_ids else 'none'}")

    # Build choices for InquirerPy
    choices = []
    for model in SUPPORTED_MODELS:
        # Build status tag
        tags = []
        if model.id in installed_ids:
            tags.append("installed")
        else:
            tags.append(f"{model.size_gb:.1f} GB")

        fit = _classify_model_fit(model.size_gb, hardware) if hardware else "unknown"
        if model.id == suggested_id:
            tags.append("suggested")
        if model.recommended:
            tags.append("recommended")
        if model.experimental:
            tags.append("experimental")
        if fit == "too_big":
            tags.append("too big")
        if fit == "ram_spill":
            tags.append("slow")

        tag_str = " | ".join(tags)
        name = f"{model.name} [{tag_str}]"

        choices.append(Choice(value=model.id, name=name))

    choices.append(Separator())
    choices.append(Choice(value="__custom__", name="Custom model path..."))

    selected = inquirer.select(
        message="Choose a model:",
        choices=choices,
        default=suggested_id if suggested_id else None,
        pointer=">",
    ).execute()

    if selected is None:
        return None

    if selected == "__custom__":
        custom_path = inquirer.filepath(
            message="Enter path to .gguf file:",
            validate=lambda x: Path(x).exists() or Path(x).expanduser().exists(),
            invalid_message="File not found",
        ).execute()

        if not custom_path:
            return await step_model_selection(hardware)

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
                return await step_model_selection(hardware)

        if not model_path.exists():
            print_error(f"File not found: {model_path}")
            return await step_model_selection(hardware)

        print_success(f"Using: {model_path.name}")
        print()
        return model_path

    # Find selected model
    model = None
    for m in SUPPORTED_MODELS:
        if m.id == selected:
            model = m
            break

    if model is None:
        return None

    model_path = models_dir / model.filename

    if model_path.exists():
        log.debug(f"Model already exists at {model_path}")
        print_success(f"Model ready: {model.name}")
        print()
    else:
        log.debug(f"Downloading model to {model_path}")
        model_path = await download_model(model, model_path)
        if model_path is None:
            return await step_model_selection(hardware)
        log.debug(f"Download complete: {model_path}")
        print_success(f"Downloaded: {model.name}")
        print()

    return model_path


async def step_nli_setup() -> bool:
    """Step 3: NLI model setup for fact verification."""
    from .nli import is_nli_model_available, download_nli_model

    print_info("Step 3/4: NLI Model Setup")
    print()

    if is_nli_model_available():
        print_success("NLI model ready: RoBERTa Large MNLI")
        print()
        return True

    console.print("[dim]NLI (Natural Language Inference) verifies NPC claims against their knowledge base.[/dim]")
    console.print("[dim]Model: RoBERTa Large MNLI (~1.4 GB)[/dim]")
    print()

    # Download with progress
    progress = ProgressDisplay(
        title="Downloading NLI Model",
        total=100,
        subtitle="RoBERTa Large MNLI from HuggingFace",
    )

    try:
        # Simple download without detailed progress (HuggingFace handles caching)
        console.print("[dim]Downloading from HuggingFace Hub...[/dim]")
        success = download_nli_model()

        if success:
            print_success("NLI model downloaded: RoBERTa Large MNLI")
            print()
            return True
        else:
            print_error("Failed to download NLI model")
            print_info("You can try again later by restarting the wizard")
            print()
            return False
    except Exception as e:
        print_error(f"Download failed: {e}")
        print()
        return False


async def run_local_chat(port: int = 8080) -> None:
    """Chat directly with llama-server (dev mode)."""
    import httpx

    console.print()
    console.rule("[bold magenta]Local Chat (Dev Mode)[/bold magenta]", style="cyan")
    console.print(f"[dim]Chatting directly with llama-server on port {port}[/dim]")
    console.print("[dim]Commands: /help /clear /quit[/dim]")
    console.print()

    history = []
    base_url = f"http://127.0.0.1:{port}"

    while True:
        try:
            user_input = inquirer.text(
                message="You:",
                mandatory=False,
            ).execute()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower().strip()

            if cmd in ("/quit", "/exit", "/back"):
                break

            if cmd == "/help":
                console.print()
                console.print("[dim]Available commands:[/dim]")
                console.print("  [cyan]/clear[/cyan]  - Clear conversation history")
                console.print("  [cyan]/quit[/cyan]   - Exit chat")
                console.print()
                continue

            if cmd == "/clear":
                history = []
                console.print("[dim]Conversation history cleared.[/dim]")
                console.print()
                continue

            console.print("[dim]Unknown command. Type /help for available commands.[/dim]")
            continue

        if user_input.lower() in ("quit", "exit"):
            break

        history.append({"role": "user", "content": user_input})

        with console.status("[dim]Thinking...[/dim]"):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        f"{base_url}/v1/chat/completions",
                        json={
                            "messages": history,
                            "max_tokens": 512,
                            "temperature": 0.7,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()

                assistant_msg = data["choices"][0]["message"]["content"]
                history.append({"role": "assistant", "content": assistant_msg})

                console.print(f"[cyan]Model:[/cyan] {assistant_msg}")
                console.print()

                if len(history) > 20:
                    history = history[-20:]

            except httpx.RequestError as e:
                print_error(f"Connection error: {e}")
            except Exception as e:
                print_error(f"Error: {e}")

    print()
    print_info("Local chat ended.")


class DownloadCancelled(Exception):
    """Raised when user cancels download."""
    pass


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
    nli_enabled: bool = True,
) -> int:
    """Step 4: Start llama-server, NLI service, and connect to backend."""
    log.debug("Starting services step")
    log.debug(f"Model: {model_path}")
    log.debug(f"Worker ID: {worker_id}")
    log.debug(f"Dev mode: {dev_mode}")
    log.debug(f"NLI enabled: {nli_enabled}")
    print_info("Step 4/4: Starting Services")
    print()

    from .llama_server import (
        LlamaServerProcess,
        is_llama_server_installed,
        download_llama_server,
        DownloadProgress,
    )

    # Use LiveStatusDisplay for startup
    status = LiveStatusDisplay(title="Loreguard Starting")
    status.start()

    # Download llama-server if needed
    if not is_llama_server_installed():
        log.debug("llama-server not installed, downloading...")
        status.set_line("server", "llama-server", "Downloading...")

        try:
            def on_progress(msg: str, prog: DownloadProgress | None):
                if prog:
                    status.set_line("server", "llama-server", f"Downloading... {int(prog.percent)}%")

            await download_llama_server(on_progress)
            status.set_line("server", "llama-server", "Downloaded")
        except Exception as e:
            status.stop()
            print_error(f"Failed to download llama-server: {e}")
            return 1

    # Start llama-server
    log.debug("Starting llama-server...")
    status.set_line("server", "llama-server", "Starting...")
    status.set_line("model", "Model", model_path.name)

    llama = LlamaServerProcess(model_path, port=8080)
    log.debug(f"llama-server command: {llama}")
    llama.start()

    log.debug("Waiting for llama-server to load model...")
    status.set_line("server", "llama-server", "Loading model...")

    ready = await llama.wait_for_ready(timeout=120.0)
    if not ready:
        log.debug("llama-server failed to start (timeout after 120s)")
        status.stop()
        print_error("llama-server failed to start (timeout)")
        llama.stop()
        return 1

    log.debug("llama-server is ready on port 8080")
    status.set_line("server", "llama-server", "Running on port 8080")

    # Connect to backend (unless dev mode)
    tunnel = None
    if not dev_mode:
        log.debug("Connecting to backend...")
        status.set_line("backend", "Backend", "Connecting...")

        try:
            from .tunnel import BackendTunnel
            from .llm import LLMProxy

            llm_proxy = LLMProxy("http://127.0.0.1:8080")
            log.debug(f"LLM proxy configured for http://127.0.0.1:8080")

            nli_service = None
            if nli_enabled:
                from .nli import NLIService
                status.set_line("nli", "NLI", "Loading model...")
                nli_service = NLIService()
                if nli_service.load_model():
                    log.debug(f"NLI service loaded (device: {nli_service.device})")
                    status.set_line("nli", "NLI", f"Ready ({nli_service.device})")
                else:
                    log.debug("NLI service failed to load")
                    status.set_line("nli", "NLI", "Failed to load")
                    nli_service = None

            model_id = _resolve_backend_model_id(model_path.stem)

            tunnel = BackendTunnel(
                backend_url="wss://api.loreguard.com/workers",
                llm_proxy=llm_proxy,
                worker_id=worker_id,
                worker_token=token,
                model_id=model_id,
                nli_service=nli_service,
            )
            log.debug(f"Backend tunnel configured: URL=wss://api.loreguard.com/workers, model_id={model_id}")

            asyncio.create_task(tunnel.connect())
            await asyncio.sleep(2)

            log.debug("Backend connection established")
            status.set_line("backend", "Backend", "Connected")
        except Exception as e:
            log.debug(f"Backend connection failed: {e}")
            status.set_line("backend", "Backend", f"Failed: {e}")
    else:
        status.set_line("mode", "Mode", "Dev (local only)")
        status.set_line("api", "API", "http://localhost:8080")

    status.stop()

    # Ask what to do next
    if dev_mode:
        choices = [
            Choice(value="local_chat", name="Chat locally - Chat directly with llama-server"),
            Choice(value="server", name="Monitor llama-server - View stats at http://localhost:8080"),
        ]
        message = "llama-server is ready. What would you like to do?"
    else:
        choices = [
            Choice(value="chat", name="Chat with NPC - Interactive chat using Loreguard API"),
            Choice(value="server", name="Monitor worker - View stats and wait for requests"),
        ]
        message = "Worker is connected to Loreguard. What would you like to do?"

    mode_choice = inquirer.select(
        message=message,
        choices=choices,
        pointer=">",
    ).execute()

    if mode_choice == "chat":
        from .npc_chat import run_npc_chat

        try:
            await run_npc_chat(api_token=token)
        except KeyboardInterrupt:
            pass

        if tunnel:
            try:
                await tunnel.disconnect()
            except:
                pass
        llama.stop()
        print_success("Goodbye!")
        return 0

    if mode_choice == "local_chat":
        await run_local_chat(port=8080)
        llama.stop()
        print_success("Goodbye!")
        return 0

    # Running state (server mode) - use LiveStatusDisplay
    live_status = LiveStatusDisplay(title="Loreguard Running")
    live_status.set_line("server", "llama-server", "Running on port 8080")
    live_status.set_line("model", "Model", model_path.name)
    if dev_mode:
        live_status.set_line("mode", "Mode", "Dev (local only)")
        live_status.set_line("api", "API", "http://localhost:8080")
    elif tunnel:
        live_status.set_line("backend", "Backend", "Connected")
    live_status.set_line("spacer", "", "")
    live_status.set_line("requests", "Requests", "0")
    live_status.set_line("tokens", "Tokens", "0")
    live_status.set_footer("Ctrl+C to stop")
    live_status.start()

    request_count = [0]
    total_tokens = [0]

    def on_request(npc: str, tokens: int, ttft_ms: float, total_ms: float):
        request_count[0] += 1
        total_tokens[0] += tokens
        tps = (tokens / total_ms * 1000) if total_ms > 0 else 0
        live_status.set_line("requests", "Requests", str(request_count[0]))
        live_status.set_line("tokens", "Tokens", f"{total_tokens[0]:,}")
        live_status.set_line("last", "Last", f"{npc} | {tokens} tok | {tps:.1f} tk/s")
        live_status.log(f"{npc}: {tokens} tokens @ {tps:.1f} tk/s", "success")

    if tunnel:
        tunnel.on_request_complete = on_request
        tunnel.log_callback = live_status.log

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

    live_status.set_title("Shutting Down")
    live_status.set_line("server", "llama-server", "Stopping...")
    live_status.set_footer("")

    llama.stop()

    if tunnel:
        try:
            await tunnel.disconnect()
        except:
            pass

    live_status.stop()
    print_success("Goodbye!")
    print()
    return 0


async def run_wizard() -> int:
    """Run the setup wizard."""
    try:
        print_banner()
        hardware = detect_hardware()
        print_info(
            f"Detected hardware: CPU {hardware.cpu} | "
            f"RAM {_format_ram_gb(hardware.ram_gb)} | "
            f"GPU {_format_gpu_info(hardware)}"
        )
        print()

        # Step 1: Authentication
        token, worker_id, dev_mode = await step_authentication()
        if token is None:
            print_error("Cancelled")
            return 1

        # Step 2: Model Selection
        model_path = await step_model_selection(hardware)
        if model_path is None:
            print_error("Cancelled")
            return 1

        # Step 3: NLI Setup
        nli_enabled = await step_nli_setup()

        # Step 4: Start
        return await step_start(model_path, token, worker_id, dev_mode, nli_enabled=nli_enabled)

    except KeyboardInterrupt:
        print()
        print_error("Interrupted")
        return 1
    finally:
        show_cursor()


def main(verbose: bool = False):
    """Entry point."""
    log_file = _configure_logging(verbose)
    if log_file:
        print(f"Debug logging to: {log_file}")
        print()
    try:
        exit_code = asyncio.run(run_wizard())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        show_cursor()
        print()
        sys.exit(1)


if __name__ == "__main__":
    import sys
    verbose = any(a in ('-v', '--verbose') for a in sys.argv[1:])
    main(verbose=verbose)
