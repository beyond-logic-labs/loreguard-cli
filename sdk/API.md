# Loreguard Client API Reference

Loreguard Client exposes a local HTTP API that any game can call. The server runs on `127.0.0.1` with a dynamic port written to `runtime.json`.

## Service Discovery

On startup, loreguard-client writes a `runtime.json` file:

| Platform | Path |
|----------|------|
| macOS | `~/Library/Application Support/loreguard/runtime.json` |
| Linux | `~/.local/share/loreguard/runtime.json` (or `$XDG_DATA_HOME/loreguard/`) |
| Windows | `%APPDATA%/loreguard/runtime.json` |

```json
{
  "port": 52341,
  "pid": 12345,
  "url": "http://127.0.0.1:52341",
  "started_at": "2026-02-20T10:30:00Z",
  "version": "0.7.0",
  "backend_connected": true
}
```

Read this file to discover the port, then make HTTP calls to `http://127.0.0.1:{port}`.

---

## Endpoints

### `GET /health`

Check if loreguard-client is running and connected to the backend.

**Response:**

```json
{
  "status": "ok",
  "backend_connected": true
}
```

Returns `500` if the server is in an error state.

---

### `GET /api/capabilities`

Discover what features this bundle supports. Use this to feature-detect before sending requests with optional fields.

**Response:**

```json
{
  "streaming": true,
  "chunk_modes": ["deberta", "sentence"],
  "manages_history": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `streaming` | bool | Whether SSE streaming is supported |
| `chunk_modes` | string[] | Available chunk detection modes (e.g. `"deberta"`, `"sentence"`) |
| `manages_history` | bool | Whether the bundle can manage conversation history internally |

---

### `POST /api/chat`

Send a player message and get an NPC response. Supports both blocking JSON and SSE streaming.

**Request Headers:**

| Header | Value | Effect |
|--------|-------|--------|
| `Content-Type` | `application/json` | Required |
| `Accept` | `text/event-stream` | Enables SSE streaming (optional) |
| `Authorization` | `Bearer <token>` | API token for character access (optional) |

**Request Body:**

```json
{
  "character_id": "merchant-npc",
  "message": "What do you have for sale?",
  "player_handle": "player1",
  "player_id": "uuid-here",
  "current_context": "player is in the marketplace",
  "scenario_id": "main-quest",
  "history": [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Welcome, traveler!"}
  ],
  "chunk_mode": "deberta",
  "manage_history": false,
  "max_speech_tokens": 150,
  "verbose": false,
  "enable_thinking": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `character_id` | string | Yes | NPC identifier |
| `message` | string | Yes | Player's message |
| `player_handle` | string | No | Player's display name |
| `player_id` | string | No | Player's unique ID for per-player state. If empty, backend uses the developer's owner ID |
| `current_context` | string | No | Game context (location, situation) |
| `scenario_id` | string | No | Scenario identifier |
| `history` | array | No | Conversation history. Omit if `manage_history` is true |
| `chunk_mode` | string | No | `"deberta"` for ML-based chunk splitting, `"sentence"` for regex sentence splitting, `""` or omit for none |
| `manage_history` | bool | No | If true, the backend manages conversation history per character+player pair |
| `max_speech_tokens` | int | No | Maximum tokens in NPC speech (0 = default) |
| `verbose` | bool | No | Include pipeline pass updates in response (for debugging) |
| `enable_thinking` | bool | No | Include NPC internal monologue |

All field names accept both `snake_case` and `camelCase` (e.g. `character_id` or `characterId`).

#### JSON Response (default)

When `Accept` is not `text/event-stream`:

```json
{
  "response": "I have potions, swords, and shields. What interests you?",
  "verified": true,
  "citations": ["knowledge/inventory.md:5"],
  "chunks": [
    "I have potions, swords, and shields.",
    "What interests you?"
  ],
  "pipeline_trace": []
}
```

| Field | Type | Description |
|-------|------|-------------|
| `response` | string | Full NPC speech |
| `verified` | bool | Whether NeMo verification passed |
| `citations` | string[] | Knowledge sources used |
| `chunks` | string[] | Sentence/chunk boundaries (only present when `chunk_mode` was set) |
| `pipeline_trace` | array | Pipeline pass details (only present when `verbose` was true) |

#### SSE Streaming Response

When `Accept: text/event-stream` is set, the response is a stream of Server-Sent Events:

```
event: filler
data: {"text": "Hmm...", "dialogueAct": "wh-question"}

event: pass_update
data: {"pass": "retrieval", "status": "complete", "latencyMs": 120}

event: token
data: {"t": "I"}

event: token
data: {"t": " have"}

event: token
data: {"t": " potions"}

event: done
data: {"speech": "I have potions...", "verified": true, "citations": [...], "chunks": [...]}

event: follow_up
data: {"speech": "By the way, new stock arrives tomorrow.", ...}
```

| Event | Data | Description |
|-------|------|-------------|
| `filler` | `{text, dialogueAct}` | Contextual filler message. Sent early (~100ms) before the pipeline completes. Display immediately for perceived responsiveness |
| `token` | `{t}` | Single token from the LLM. Append to build the response incrementally |
| `pass_update` | `{pass, status, ...}` | Pipeline pass progress (only when `verbose` is true). For debugging |
| `done` | `{speech, verified, citations, chunks, ...}` | Final verified response. Contains the same fields as the JSON response |
| `follow_up` | `{speech, ...}` | Unsolicited follow-up message from the NPC (may arrive after `done`) |
| `error` | `{error}` | Error message. Stream ends after this |

---

## Integration Patterns

### Minimal (any language)

Just POST JSON and read the response:

```
POST http://127.0.0.1:{port}/api/chat
Content-Type: application/json

{"character_id": "merchant", "message": "Hello!"}
```

### With Chunks (for staggered display)

Request chunked responses and display each chunk separately with delays:

```json
{
  "character_id": "merchant",
  "message": "Hello!",
  "chunk_mode": "deberta"
}
```

Response includes `chunks` array. Display each chunk as a separate message/bubble with ~500-700ms delays between them.

### With Server-Managed History

Let the backend track conversation history so your game doesn't have to:

```json
{
  "character_id": "merchant",
  "message": "Hello!",
  "player_id": "player-uuid",
  "manage_history": true
}
```

No `history` field needed. The backend maintains a rolling conversation buffer per `(character_id, player_id)` pair.

### With Streaming + Filler

For games with real-time UX (speech bubbles, typing indicators):

1. Set `Accept: text/event-stream`
2. On `filler` event: show typing indicator or filler text ("Hmm...")
3. On `token` events: append tokens to display
4. On `done` event: finalize the response, show verified status

---

## SDK Files

Pre-built SDK files for common engines are in the `sdk/` directory. Copy the relevant file into your project:

| Engine | File | Notes |
|--------|------|-------|
| Python | `sdk/python/loreguard_sdk.py` | Requires `httpx`. Async + sync support |
| JavaScript / Electron | `sdk/javascript/loreguard-sdk.js` | Node.js CommonJS. Uses `fetch` |
| Unity / C# | `sdk/csharp/LoreguardSDK.cs` | Uses `UnityWebRequest`. Coroutine-based |
| Godot 4 | `sdk/gdscript/LoreguardSDK.gd` | Signal-based. Supports streaming |

These are thin HTTP wrappers around the endpoints documented above. You can also call the API directly from any language that supports HTTP.
