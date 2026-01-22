# Third-Party Notices

Loreguard CLI downloads and uses third-party machine learning models during setup.
This document provides attribution and license information for these components.

## Models Downloaded at Runtime

The following models may be downloaded during Loreguard setup. Downloads are cached
locally in standard directories (`~/.cache/huggingface/` for transformer models,
`~/.loreguard/models/` for GGUF models).

---

## NLI & Classification Models (HuggingFace Transformers)

These models are downloaded automatically when using citation verification and
intent classification features.

### HHEM-2.1-Open (Citation Verification)

- **Model ID**: `vectara/hallucination_evaluation_model`
- **Source**: https://huggingface.co/vectara/hallucination_evaluation_model
- **License**: Apache 2.0
- **Author**: Vectara
- **Size**: ~0.6 GB
- **Purpose**: Hallucination evaluation / grounding for verifying NPC claims against knowledge base

### DeBERTa-v3-large-zeroshot (Intent Classification)

- **Model ID**: `MoritzLaurer/DeBERTa-v3-large-zeroshot-v2.0`
- **Source**: https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-zeroshot-v2.0
- **License**: MIT (foundation model)
- **Author**: Moritz Laurer
- **Size**: ~800 MB
- **Purpose**: Zero-shot classification for adaptive retrieval strategy

**Note**: The foundation model is MIT licensed. Training data includes datasets with
varying licenses. For commercial use with strict license requirements, consider the
`-c` variants which use only commercially-friendly training data.

---

## LLM Models (GGUF Format)

These models are offered for download during the wizard. Users select one model
based on their hardware capabilities.

### Qwen3 Models (Apache 2.0)

| Model | Size | License | Source |
|-------|------|---------|--------|
| Qwen3-4B-Instruct | 2.8 GB | Apache 2.0 | [HuggingFace](https://huggingface.co/lmstudio-community/Qwen3-4B-Instruct-2507-GGUF) |
| Qwen3-8B | 5.0 GB | Apache 2.0 | [HuggingFace](https://huggingface.co/Qwen/Qwen3-8B-GGUF) |
| Qwen3-1.7B | 1.1 GB | Apache 2.0 | [HuggingFace](https://huggingface.co/unsloth/Qwen3-1.7B-GGUF) |

- **Organization**: Alibaba Cloud / Qwen Team
- **License**: Apache License 2.0
- **License URL**: https://www.apache.org/licenses/LICENSE-2.0
- **Project**: https://github.com/QwenLM/Qwen3

### Llama Models (Llama Community License)

| Model | Size | License | Source |
|-------|------|---------|--------|
| Llama-3.2-3B-Instruct | 2.0 GB | Llama 3.2 Community | [HuggingFace](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) |
| Meta-Llama-3-8B-Instruct | 4.9 GB | Llama 3 Community | [HuggingFace](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF) |

- **Organization**: Meta Platforms, Inc.
- **License**: Llama 3.2 Community License / Llama 3 Community License
- **License URL**: https://www.llama.com/llama3_2/license/
- **Attribution**: Built with Llama

**Llama License Notes**:
- Non-exclusive, worldwide, royalty-free license for use, reproduction, and distribution
- Requires attribution notice and "Built with Llama" display
- Commercial users with >700M monthly active users must request additional license from Meta
- Full license terms: https://www.llama.com/llama3_2/license/

### RNJ-1 (Apache 2.0)

| Model | Size | License | Source |
|-------|------|---------|--------|
| RNJ-1-Instruct | 6.1 GB | Apache 2.0 | [HuggingFace](https://huggingface.co/lmstudio-community/rnj-1-instruct-GGUF) |

- **Organization**: EssentialAI
- **Original Model**: https://huggingface.co/EssentialAI/rnj-1-instruct
- **GGUF Quantization**: LM Studio team
- **License**: Apache License 2.0

### GPT-OSS-20B (Apache 2.0)

| Model | Size | License | Source |
|-------|------|---------|--------|
| GPT-OSS-20B | 11.5 GB | Apache 2.0 | [HuggingFace](https://huggingface.co/lmstudio-community/gpt-oss-20b-GGUF) |

- **Organization**: OpenAI
- **Original Model**: https://huggingface.co/openai/gpt-oss-20b
- **GGUF Quantization**: LM Studio team
- **License**: Apache License 2.0

---

## Runtime Dependencies

### llama.cpp / llama-server

- **Project**: https://github.com/ggerganov/llama.cpp
- **License**: MIT
- **Author**: Georgi Gerganov and contributors
- **Purpose**: Local LLM inference server

Loreguard downloads pre-built `llama-server` binaries for the user's platform.

---

## License Texts

### Apache License 2.0

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### MIT License

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Custom Models

Users may also provide their own `.gguf` model files. When using custom models,
users are responsible for ensuring compliance with the respective model's license.

---

## Questions

For questions about third-party licenses or attributions, please open an issue at:
https://github.com/beyond-logic-labs/loreguard-cli/issues
