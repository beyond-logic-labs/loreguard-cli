# Loreguard

[![PyPI version](https://img.shields.io/pypi/v/loreguard-cli.svg)](https://pypi.org/project/loreguard-cli/)
[![Build](https://github.com/beyond-logic-labs/loreguard-cli/actions/workflows/release.yml/badge.svg)](https://github.com/beyond-logic-labs/loreguard-cli/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub release](https://img.shields.io/github/v/release/beyond-logic-labs/loreguard-cli)](https://github.com/beyond-logic-labs/loreguard-cli/releases)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                                                                                │
│  ██╗      ██████╗ ██████╗ ███████╗   ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗  │
│  ██║     ██╔═══██╗██╔══██╗██╔════╝  ██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗ │
│  ██║     ██║   ██║██████╔╝█████╗    ██║  ███╗██║   ██║███████║██████╔╝██║  ██║ │
│  ██║     ██║   ██║██╔══██╗██╔══╝    ██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║ │
│  ███████╗╚██████╔╝██║  ██║███████╗  ╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝ │
│  ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝   ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝  │
│                                                                                │
│  Local inference for your game NPCs                                            │
│  loreguard.com                                                                 │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

AI-Powered NPCs using your own hardware (your servers or your player's hardware) 
Loreguard CLI connects the LLM Inference to the Loreguard NPC system.

## How It Works

```
┌─────────────────┐    wss://api.loreguard.com    ┌─────────────────┐
│   Your Game     │◄────────────────────────────► │  Loreguard API  │
│  (NPC Dialog)   │                               │    (Backend)    │
└─────────────────┘                               └────────┬────────┘
                                                           │
                                                           │ Routes inference
                                                           │ to your worker
                                                           ▼
                                                  ┌─────────────────┐
                                                  │  Loreguard CLI  │◄── You run this
                                                  │  (This repo)    │
                                                  └────────┬────────┘
                                                           │
                                                           │ Local inference
                                                           ▼
                                                  ┌─────────────────┐
                                                  │   llama.cpp     │
                                                  │  (Your GPU/CPU) │
                                                  └─────────────────┘
```

## Installation

### Option 1: Download Binary (Recommended)

Download standalone binaries from [Releases](https://github.com/beyond-logic-labs/loreguard-cli/releases):
- `loreguard-linux` - Linux x64
- `loreguard-macos` - macOS (Intel & Apple Silicon)
- `loreguard-windows.exe` - Windows x64

### Option 2: Install from PyPI

```bash
pip install loreguard-cli
```

### Option 3: Install from Source

```bash
git clone https://github.com/beyond-logic-labs/loreguard-cli
cd loreguard-cli
pip install -e .
```

### Option 4: Build Your Own Binary

```bash
git clone https://github.com/beyond-logic-labs/loreguard-cli
cd loreguard-cli
pip install -e ".[build]"
python scripts/build.py
# Output: dist/loreguard (or dist/loreguard.exe on Windows)
```

## Quick Start

### Interactive Wizard

```bash
loreguard
```

The wizard guides you through:
1. **Authentication** - Enter your worker token
2. **Model Selection** - Choose or download a model
3. **Running** - Starts llama-server and connects to backend

### Headless CLI

```bash
loreguard-cli --token lg_worker_xxx --model /path/to/model.gguf
```

Or auto-download a supported model:

```bash
loreguard-cli --token lg_worker_xxx --model-id qwen3-4b-instruct
```

**Environment variables:**
```bash
export LOREGUARD_TOKEN=lg_worker_xxx
export LOREGUARD_MODEL=/path/to/model.gguf
loreguard-cli
```

## Supported Models

| Model ID | Name | Size | Notes |
|----------|------|------|-------|
| `qwen3-4b-instruct` | Qwen3 4B Instruct | 2.8 GB | **Recommended** |
| `llama-3.2-3b-instruct` | Llama 3.2 3B | 2.0 GB | Fast |
| `qwen3-8b` | Qwen3 8B | 5.2 GB | Higher quality |
| `meta-llama-3-8b-instruct` | Llama 3 8B | 4.9 GB | General purpose |

Or use any `.gguf` model with `--model /path/to/model.gguf`.

## Use Cases

### For Game Developers (Testing & Development)

Use Loreguard CLI during development to test NPC dialogs with your own hardware:

```bash
# Start the worker
loreguard-cli --token $YOUR_DEV_TOKEN --model-id qwen3-4b-instruct

# Your game connects to Loreguard API
# NPC inference requests are routed to your local worker
```

### For Players (Coming Soon)

> **Note:** Player distribution support is in development. Currently, players would need their own Loreguard account and token.

We're working on a **Game Keys** system that will allow:
- Developers to register their game and get a Game API Key
- Players to run the CLI without needing a Loreguard account
- Automatic worker provisioning scoped to each game

**Interested in early access?** Contact us at [loreguard.com](https://loreguard.com)

## Requirements

- **RAM**: 8GB minimum (16GB+ for larger models)
- **GPU**: Optional but recommended (NVIDIA CUDA or Apple Silicon)
- **Disk**: 2-6GB depending on model
- **Python**: 3.10+ (if installing from source)

## Get Your Token

1. Go to [loreguard.com/developers](https://loreguard.com/developers)
2. Create a worker token
3. Use it with `--token` or `LOREGUARD_TOKEN`

## Development

```bash
git clone https://github.com/beyond-logic-labs/loreguard-cli
cd loreguard-cli
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Run interactive wizard
python -m src.wizard

# Run headless CLI
python -m src.cli --help

# Run tests
pytest
```

## License

MIT
