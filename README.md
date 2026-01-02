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

### Linux / macOS

```bash
pip install loreguard-cli
```

### Windows

Download `loreguard.exe` from [Releases](https://github.com/beyond-logic-labs/loreguard-cli/releases).

Or install via pip if you have Python:

```bash
pip install loreguard-cli
```

### From Source

```bash
git clone https://github.com/beyond-logic-labs/loreguard-cli
cd loreguard-cli
pip install -e .
```

## Quick Start

### Interactive Mode (no arguments)

```bash
loreguard
```

The wizard guides you through:
1. **Authentication** - Enter your worker token
2. **Model Selection** - Choose or download a model
3. **Running** - Starts llama-server and connects to backend

### Headless Mode (with arguments)

```bash
loreguard --token lg_worker_xxx --model /path/to/model.gguf
```

Or auto-download a supported model:

```bash
loreguard --token lg_worker_xxx --model-id qwen3-4b-instruct
```

**Environment variables:**
```bash
export LOREGUARD_TOKEN=lg_worker_xxx
export LOREGUARD_MODEL=/path/to/model.gguf
loreguard
```

### Chat Mode (test NPC pipeline)

Test your NPC chat without running a local model:

```bash
loreguard --chat --token lg_worker_xxx
```

This connects directly to the Loreguard API to:
- List your registered NPCs
- Select one to chat with
- See verification status and latency

## Supported Models

Works with any `.gguf` model. Tested with the following model families:

- **Qwen** - Recommended for best quality/speed balance
- **Llama** - Meta's open models
- **GPT** - GPT-style open models
- **RNJ** - Specialized models
- **Violet Lotus** - Community fine-tunes

Use any model with `--model /path/to/model.gguf`.

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
