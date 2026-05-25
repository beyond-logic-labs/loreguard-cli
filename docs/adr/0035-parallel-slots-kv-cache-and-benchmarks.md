# ADR-0035: Parallel Slots, KV Cache Type, and Local-Inference Benchmarks

## Status

Accepted

## Context

`loreguard-cli` runs `llama-server` on the end-user's GPU (and on capable nodes
serving multiple sessions). Until now the server was launched with a **single forced slot**
(`-np 1`, ADR-0014) and **f16 KV cache**. That is the right default for a typical
player PC serving one session, but it leaves capable GPUs unable to serve
multiple concurrent sessions.

We needed to answer, with real measurements on consumer hardware:

1. Is llama.cpp or vLLM the right engine for this workload?
2. What actually limits concurrency, and how many concurrent sessions fit?
3. Does prompt caching (the ADR-0014 cognitive-context reuse) hold up under
   concurrency, and what does it buy?
4. Which KV cache type should we default to?
5. What is the real end-to-end latency of a full multi-pass NPC turn, with and
   without prompt caching?
6. Which model gives the best quality/grounding per token — the 26B MoE (Gemma)
   or a dense 14B (Qwen3)?

All benchmarks below were run on **RTX 5080 (16 GB, Blackwell sm_120), WSL2 /
Ubuntu, CUDA 13.2 llama.cpp build (`d161ea7`)**, using the model we target,
`gemma-4-26B-A4B-it` (MoE, sliding-window attention) at `UD-Q3_K_XL` (~13 GB),
plus `Qwen3-14B` (dense, full attention) for the engine comparison. The Gemma
runs were launched through `LlamaServerProcess` itself (this CLI), so the numbers
reflect our actual launch path.

## Decision

Add two server/dev knobs to `LlamaServerProcess` (env-configurable, see
`.env.example`):

- **`LOREGUARD_PARALLEL_SLOTS`** (default `1`) → `-np N`, with `-c` scaled to
  `context_size * N` so each slot keeps the full context window.
- **`LOREGUARD_KV_CACHE_TYPE`** (default **`q8_0`**) → `-ctk/-ctv` + `-fa on`.
  Quantized KV is what lets multiple slots fit in VRAM.

Per-request `id_slot` (in `LLMRequest`) lets the caller map session → slot.

**Default KV type is `q8_0`, not a smaller quant — see Table 4 (lower-bit KV
produced gibberish on this model).**

## Benchmarks

### Table 1 — Engine batch throughput (Qwen3-14B, 4-bit, 256-tok outputs)

Aggregate generation throughput vs concurrency, identical OpenAI client.

| Concurrency | vLLM (AWQ) tok/s | llama.cpp (Q4_K_M) tok/s | vLLM advantage |
|---:|---:|---:|:---:|
| 1 | 88 | 84 | tie |
| 4 | 346 | 257 | 1.35x |
| 8 | 678 | 317 | 2.14x |
| 16 | 1264 | 753 | 1.68x |
| 32 | 2259 | 1105 | 2.04x |

p99 latency @ conc 32: vLLM 3.7 s vs llama.cpp 7.7 s.

**Takeaway:** vLLM wins batch (~2x). **But** vLLM is CUDA-only, needs models that
fit its full-precision-embedding quants, and cannot run our 26B Gemma on 16 GB
(all vLLM-native quants are ≥17 GB). For a P2P network of heterogeneous consumer
GPUs, **llama.cpp (GGUF) is the portable runtime** and is the engine `loreguard-cli`
ships. The parallel-slots work narrows the gap on capable nodes.

### Table 2 — KV cache is the concurrency limit (Qwen3-14B, vLLM, 0.90 util)

Total KV pool = **21,680 tokens** (≈3.3 GiB left after 14B weights). Max
concurrent sessions ≈ pool ÷ tokens-per-session:

| Tokens / session | Max concurrent |
|---:|:---:|
| 256 | ~60 |
| 4,096 | 5.3 |
| 8,192 | 2.6 |
| 16,384 | **1.17** |

A full-attention 14B at 16k context serves ~1 user on 16 GB. This is why the
model architecture (next table) matters more than parameter count.

### Table 3 — Gemma-4-26B-A4B (SWA) slot capacity + prompt cache (q8_0 KV, via loreguard-cli)

Sliding-window attention makes per-session KV tiny, so the 26B fits many 16k-class
sessions where the 14B fits one. Measured, stable under a 10-way concurrent burst:

| Prefill / session | Stable slots | VRAM @ load | Turn-1 prefill (cache MISS) | Turn-2 prefill (cache HIT) | Turn-2 speedup |
|---:|:---:|---:|---:|---:|:---:|
| ~4,096 tok | **10** | 15.6 / 16.3 GB | 4,595 tok | **24 tok** | 4.5x |
| ~8,192 tok | **8** | 15.5 / 16.3 GB | 9,155 tok | **24 tok** | ~7x |

**Prompt caching works under concurrency:** with `cache_prompt: true` + a pinned
`id_slot` per session, turn 2 reuses the cognitive-context KV and prefills only the
new player line (~24 tokens) instead of re-processing 4–9k. This is the ADR-0014
optimization holding across 8–10 concurrent sessions.

**Capacity vs burst:** KV capacity allows ~10–16 *resident* sessions, but the
GPU can only **cold-prefill ~10 of them simultaneously** (activation/compute
buffers, not KV, bind at the top). In production this is fine: with caching, cold
prefills are rare (first turn only); steady state is cache hits. Rule of thumb on
16 GB: `LOREGUARD_PARALLEL_SLOTS=8` (8k contexts) to `10` (4k), leaving compute
headroom. "Loads at N slots" ≠ "stable under an N-way cold-prefill burst" — size
~2 slots below the load ceiling.

### Table 4 — KV cache type vs output quality (Gemma Q3_K_XL, single request)

Same prompt and sampling; only `-ctk/-ctv` changed:

| KV type | Output | Verdict |
|---|---|:---:|
| **q8_0** | "the ocean covers approximately 71% of the…" — coherent | ✅ use |
| q5_1 | "ocean is a deep blue ในการ- ในการ-…" — degrades to gibberish | ❌ |
| q5_0 | "a vast-vast-vast-vast…" — repetition collapse | ❌ |
| q4_0 | "<\|channel\|>The ocean covers 70%…" — stray tokens | ⚠️ |

**On this model, lower-bit KV breaks coherence** (already-aggressive Q3 weights +
Gemma's known quantization sensitivity). The generic "q5_1 KV is near-lossless"
rule does **not** hold here. q8_0 is the only quantized KV that stays coherent —
hence the `q8_0` default. (KV type is configurable, so other models can opt into
smaller KV if they tolerate it.)

### Table 5 — Real multi-pass NPC turn latency (Gemma-26B, q8_0, real engine prompts)

A turn = Pass 2 (think) + Pass 4 (speak) + Pass 4.5 (review), run via the engine's
`cmd/pipeline` against `llama-server`, character "zero" (~750-token cognitive context),
realistic per-pass token caps (256 / 50 / 80):

| | 1 user | 10 simultaneous |
|---|---:|---:|
| Pass 2 (think, 256 tok) | 2.5 s | 12.0 s |
| Pass 4 (speak, 50 tok) | 0.9 s | 7.1 s |
| Pass 4.5 (review, 80 tok) | 1.2 s | 6.2 s |
| **Full turn** | **4.5 s** | **~25 s** (p50≈p90) |

The 10-simultaneous case is **decode-contention bound** (~19 tok/s/user at 10-way) —
prompt caching does **not** fix this (caching saves prefill, not decode). 10 truly
concurrent is worst-case; real effective concurrency is lower (users think between
turns), so ~3–5 active gives ~8–12 s/turn.

### Table 6 — Multi-turn WITH prompt caching (single user, 6-turn thread)

`-np 1` (llama.cpp auto prefix caching), streaming for TTFT:

| Metric | Gemma-4-26B (Q3 SWA) | Qwen3-14B (Q4 dense) |
|---|---:|---:|
| Decode speed | **~130 tok/s** | ~84 tok/s |
| TTFT cold (turn 1) | 303 ms | 274 ms |
| TTFT cached (turn 2+) | **~110 ms** | ~60 ms |
| TTFT p50 / p90 | 110 / 303 ms | 60 / 274 ms |
| `cached_tokens` over turns | 0 → 792 → 1072 ✅ | 0 → 775 → 1046 ✅ |

**Caching confirmed:** `cached_tokens` jumps on turn 2+ and TTFT drops ~3–5x vs the
cold turn 1 — llama.cpp reuses the cached prefix automatically (no flag). Here the
context is small (~800 tok) so the saving is ~200 ms/turn; at the target **4–16k
contexts it is seconds/turn**. **Gemma decodes faster than the dense 14B (130 vs 84
tok/s)** because it is a 4B-active MoE (sparsity), despite being "26B". With caching a
Gemma NPC turn is **sub-second single-user** (TTFT ~110 ms + ~30–50 tok @ 130 tok/s).

### Table 7 — Model quality / grounding (6-turn FAIRVIEW thread, character "zero")

| | Gemma-4-26B | Qwen3-14B |
|---|---|---|
| Voice (lowercase, terse, no metaphor) | ✅ | ⚠️ used a metaphor ("it's a door") |
| Trust-gated secrets | ✅ deflected personal Qs; never named the gated secret | ❌ leaked "Phantom" (inner-circle) to a low-trust recruit |
| Grounding / memory | ✅ recalled a working-memory detail ("the relay") | — |
| Hallucinated facts | none | none |

**Gemma is better on quality AND faster** — it respected voice and trust-gated
knowledge (the product's "NPCs only say what they know" premise) while Qwen leaked a
gated character secret. **Recommendation: Gemma-4-26B-A4B is the default model;**
Qwen-14B is only a faster-TTFT fallback where the 26B can't fit.

### Operational gotchas

- **Gemma 4 has thinking ON by default** → `/v1/chat/completions` returns empty
  `content` (the text goes to `reasoning_content`) unless the request sends
  `chat_template_kwargs:{enable_thinking:false}`. Keep thinking disabled for the pipeline.
- **q5_1 / q4_0 KV degrade Gemma to gibberish** (Table 4) → keep `q8_0`.
- **`-np` must match real concurrency for MoE:** with `-np 32` but only ~10 active,
  Gemma throughput collapsed (MoE expert-scatter under a large slot pool). Set
  `LOREGUARD_PARALLEL_SLOTS` to the *expected active* concurrency, not a max.

## Consequences

- Default behavior unchanged for existing single-session installs (`-np 1`), but
  KV is now `q8_0`+flash-attention by default (near-lossless, smaller, faster).
- Capable GPUs set `LOREGUARD_PARALLEL_SLOTS` to serve multiple
  sessions; `id_slot` routes each session to its slot, preserving per-session
  cognitive-context caching.
- **Follow-up (orchestrator):** a session→slot LRU pool that assigns live sessions
  to slots and uses the existing `/slots` save/restore to park idle sessions on
  disk (avoids re-prefilling a returning user's 4–9k context). This CLI now
  provides the server capacity + `id_slot`; the pool logic lives in the
  calling client / dispatcher.
- Slot count should be chosen per GPU VRAM (and KV type); auto-detection from
  `hardware_info` is a possible future enhancement.

### Ready for P2P worker deployment

A 16 GB node serving the NPC pipeline should run:

```
LOREGUARD_PARALLEL_SLOTS=4   # 4–8 active sessions; raise only with spare VRAM/compute
LOREGUARD_KV_CACHE_TYPE=q8_0 # keep q8_0 (lower bits = gibberish on Gemma)
# model: gemma-4-26B-A4B-it UD-Q3_K_XL ; thinking disabled (enable_thinking=false)
```

This gives ~4.5 s/turn single-user and sub-second cached turns, holding Gemma's
quality/grounding edge. The one remaining piece for a multi-session worker is the
**session→slot LRU dispatcher** (assign live sessions to the N slots, `id_slot` per
request, and `/slots` save/restore to park idle sessions on disk) — that lives in
the calling client / dispatcher, not in this CLI.
