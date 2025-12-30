# Loreguard

Local inference client for Loreguard NPCs. Run AI-powered NPCs on your own hardware.

## Installation

```bash
pip install loreguard
```

Or download standalone binaries from [Releases](https://github.com/beyond-logic-labs/loreguard-cli/releases).

## Usage

### Option 1: Interactive Wizard

```bash
loreguard
```

Simple terminal wizard that guides you through:
1. **Authentication** - Enter your worker token (or type `dev` for dev mode)
2. **Model Selection** - Choose or download a model
3. **Running** - Starts llama-server and connects to backend

Works on any terminal (Windows cmd, PowerShell, bash, zsh, etc.)

### Option 2: CLI (Headless for Games)

Perfect for shipping with your game:

```bash
loreguard-cli --token lg_worker_xxx --model /path/to/model.gguf
```

Or auto-download a supported model:

```bash
loreguard-cli --token lg_worker_xxx --model-id qwen3-4b-instruct
```

**Environment variables (alternative to args):**
```bash
export LOREGUARD_TOKEN=lg_worker_xxx
export LOREGUARD_MODEL=/path/to/model.gguf
# or
export LOREGUARD_MODEL_ID=qwen3-4b-instruct

loreguard-cli
```

**Example output:**
```
2024-01-15 14:32:05 [INFO] ================================================
2024-01-15 14:32:05 [INFO] Loreguard CLI - Starting
2024-01-15 14:32:05 [INFO] ================================================
2024-01-15 14:32:05 [INFO] Using model: /models/qwen3-4b.gguf
2024-01-15 14:32:06 [INFO] Starting llama-server on port 8080...
2024-01-15 14:32:10 [INFO] llama-server ready
2024-01-15 14:32:10 [INFO] Connecting to wss://api.loreguard.com/workers...
2024-01-15 14:32:12 [INFO] Backend connection established
2024-01-15 14:32:12 [INFO] ================================================
2024-01-15 14:32:12 [INFO] Ready! Waiting for inference requests...
2024-01-15 14:32:12 [INFO] Press Ctrl+C to stop
2024-01-15 14:32:12 [INFO] ================================================
2024-01-15 14:32:45 [INFO] Request #1: Merchant | 156 tokens | 423ms TTFT | 3.2s total | 48.7 tk/s
2024-01-15 14:33:02 [INFO] Request #2: Guard | 89 tokens | 312ms TTFT | 1.8s total | 49.4 tk/s
```

## Supported Models

| Model ID | Name | Size | Notes |
|----------|------|------|-------|
| `qwen3-4b-instruct` | Qwen3 4B Instruct | 2.8 GB | **Recommended** |
| `llama-3.2-3b-instruct` | Llama 3.2 3B | 2.0 GB | Fast |
| `qwen3-8b` | Qwen3 8B | 5.2 GB | Higher quality |
| `meta-llama-3-8b-instruct` | Llama 3 8B | 4.9 GB | General purpose |

Or use any `.gguf` model with `--model /path/to/model.gguf`.

## Embedding in Your Game

For shipping with your game, use the CLI mode:

```bash
# In your game's startup script
./loreguard-cli --token $PLAYER_TOKEN --model ./models/npc-model.gguf &
```

Or bundle the Python package:
```python
from src.cli import LoreguardCLI
import asyncio

cli = LoreguardCLI(
    token="lg_worker_xxx",
    model_path=Path("./models/model.gguf"),
)
asyncio.run(cli.run())
```

## Requirements

- **RAM**: 8GB minimum (16GB+ for larger models)
- **GPU**: Optional but recommended (NVIDIA CUDA or Apple Silicon)
- **Disk**: 2-6GB depending on model
- **Python**: 3.10+ (for pip install)

## Get Your Token

1. Go to [loreguard.com/developers](https://loreguard.com/developers)
2. Create a worker token
3. Use it with `--token` or `LOREGUARD_TOKEN`

## Development

```bash
git clone https://github.com/beyond-logic-labs/loreguard-cli
cd loreguard-cli
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run TUI
python -m src.tui.app

# Run CLI
python -m src.cli --help
```

## License

MIT
