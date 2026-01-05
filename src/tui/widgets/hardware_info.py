"""Hardware info widget displaying CPU, RAM, and GPU info."""

import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

from textual.widgets import Static
from rich.text import Text
from rich.style import Style

from ..styles import FG, FG_DIM


@dataclass
class HardwareData:
    """Hardware information data class."""

    cpu: str
    ram_gb: Optional[float]
    gpu: str
    gpu_vram_gb: Optional[float]
    gpu_mem_type: Optional[str]


def _run_cmd(args: list[str]) -> str:
    """Run a command and return stdout."""
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False, timeout=3)
        return result.stdout.strip()
    except Exception:
        return ""


def _get_cpu_name() -> str:
    """Detect CPU name."""
    cpu = platform.processor().strip()
    if cpu:
        return cpu
    if sys.platform == "darwin":
        cpu = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return cpu or "Unknown"


def _get_ram_gb() -> Optional[float]:
    """Detect total RAM in GB."""
    if sys.platform == "darwin":
        mem = _run_cmd(["sysctl", "-n", "hw.memsize"])
        if mem.isdigit():
            return round(int(mem) / (1024**3), 1)
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            return round(int(parts[1]) * 1024 / (1024**3), 1)
        except Exception:
            pass
    return None


def _get_gpu_info() -> tuple[str, Optional[float], Optional[str]]:
    """Detect GPU name, VRAM, and memory type."""
    if sys.platform == "darwin":
        output = _run_cmd(["system_profiler", "SPDisplaysDataType"])
        names = []
        vram = None
        mem_type = None
        for line in output.splitlines():
            if "Chipset Model:" in line:
                names.append(line.split(":", 1)[1].strip())
            if "VRAM" in line and ":" in line:
                try:
                    val = line.split(":", 1)[1].strip().split()[0]
                    vram = float(val) if val.replace(".", "").isdigit() else None
                except Exception:
                    pass
            if "Memory Type:" in line:
                mem_type = line.split(":", 1)[1].strip()
        return ", ".join(names) if names else "Unknown", vram, mem_type
    return "Unknown", None, None


def detect_hardware() -> HardwareData:
    """Detect all hardware info."""
    cpu = _get_cpu_name()
    ram_gb = _get_ram_gb()
    gpu, vram, mem_type = _get_gpu_info()
    return HardwareData(cpu=cpu, ram_gb=ram_gb, gpu=gpu, gpu_vram_gb=vram, gpu_mem_type=mem_type)


class HardwareInfo(Static):
    """Widget displaying hardware info and connection status in a single line."""

    def __init__(self) -> None:
        super().__init__()
        self._hardware: Optional[HardwareData] = None
        self._connection_status: str = ""  # "", "connecting", "connected", "disconnected"

    def on_mount(self) -> None:
        """Detect hardware when widget is mounted."""
        self._hardware = detect_hardware()
        self.refresh()

    def set_connection_status(self, status: str) -> None:
        """Update the connection status and refresh."""
        self._connection_status = status
        self.refresh()

    def render(self) -> Text:
        """Render the hardware info line."""
        if not self._hardware:
            return Text("")

        text = Text()

        # CPU
        text.append("CPU: ", style=Style(color=FG_DIM))
        text.append(self._hardware.cpu, style=Style(color=FG))

        # RAM
        if self._hardware.ram_gb:
            text.append("   RAM: ", style=Style(color=FG_DIM))
            text.append(f"{self._hardware.ram_gb:.0f}GB", style=Style(color=FG))

        # GPU
        if self._hardware.gpu and self._hardware.gpu != "Unknown":
            text.append("   GPU: ", style=Style(color=FG_DIM))
            text.append(self._hardware.gpu, style=Style(color=FG))

        # Connection status (right side)
        if self._connection_status:
            text.append("   ")
            if self._connection_status == "connected":
                text.append("●", style=Style(color="#50FA7B", bold=True))  # Green
                text.append(" Backend", style=Style(color=FG_DIM))
            elif self._connection_status == "connecting":
                text.append("○", style=Style(color="#F1FA8C", bold=True))  # Yellow
                text.append(" Connecting...", style=Style(color=FG_DIM))
            elif self._connection_status == "disconnected":
                text.append("●", style=Style(color="#FF5555", bold=True))  # Red
                text.append(" Offline", style=Style(color=FG_DIM))
            elif self._connection_status == "dev":
                text.append("●", style=Style(color="#BD93F9", bold=True))  # Purple
                text.append(" Dev Mode", style=Style(color=FG_DIM))

        return text

    @property
    def hardware(self) -> Optional[HardwareData]:
        """Get the detected hardware data."""
        return self._hardware
