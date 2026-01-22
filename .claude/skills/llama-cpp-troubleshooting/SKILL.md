---
name: llama-cpp-troubleshooting
description: |
  Troubleshoot common llama.cpp server issues including: (1) Slot save/restore returning
  HTTP 500 errors - requires --slots flag, (2) JSON structured output causing model hangs
  or stalls - grammar blocks EOS token, (3) response_format json_schema type not working -
  use json_object with schema field instead, (4) All outputs becoming JSON function calls -
  chat template auto-detection issue. Use when: llama-server returns 500 on /slots
  endpoint, LLM stops generating mid-response, JSON parse errors from truncated output,
  "Stream stalled - no token received" errors, all outputs are JSON even without forceJsonOutput.
author: Claude Code
version: 1.1.0
date: 2026-01-21
---

# llama.cpp Server Troubleshooting

## Problem 1: Slot Save/Restore Returns HTTP 500

### Symptoms
- `Slot restore failed: 500` or `Slot save failed: 500` in logs
- KV cache slot operations failing silently
- `/slots/{id}?action=save` or `/slots/{id}?action=restore` returns 500

### Root Cause
The `/slots` API endpoint is **disabled by default** for security reasons. The `--slot-save-path`
flag alone is not sufficient.

### Solution
Add **both** flags when starting llama-server:

```bash
llama-server --slots --slot-save-path /path/to/cache ...
```

- `--slots` - Enables the `/slots` API endpoint (required!)
- `--slot-save-path` - Directory where KV cache files are stored

### Verification
After restart, slot operations should return 200 with token counts instead of 500.

---

## Problem 2: LLM Hangs/Stalls During JSON Generation

### Symptoms
- "Stream stalled - no token received for inactivity period"
- Model generates some tokens then stops for 60+ seconds
- `finish_reason: "length"` instead of `"stop"`
- JSON parse errors from truncated output
- Hundreds of newlines generated instead of stopping

### Root Cause
When using `response_format` for JSON output, llama.cpp uses a grammar constraint that:
1. Allows trailing whitespace/newlines as valid JSON
2. Blocks the EOS (end-of-sequence) token because it's not part of the grammar
3. Model keeps generating valid-but-useless whitespace until hitting max_tokens

### Solution
Use `json_object` type with explicit `schema` field (not `json_schema` type):

```python
# WORKS - json_object with schema
payload["response_format"] = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {...},
        "required": [...]
    }
}

# BROKEN - json_schema type (PR #18963 pending fix)
payload["response_format"] = {
    "type": "json_schema",
    "json_schema": {"schema": {...}}
}
```

The schema provides tighter grammar constraints that help prevent infinite whitespace generation.

### Additional Mitigations
1. Check `finish_reason` in response - if `"length"`, output was truncated
2. Increase `max_tokens` for JSON responses
3. Add explicit stop sequences for JSON closing patterns

---

## Problem 3: response_format json_schema Not Working

### Symptoms
- Error: "response_format type must be one of 'text' or 'json_object'"
- `json_schema` type silently ignored
- Schema not enforced, model outputs any JSON

### Root Cause
llama.cpp's `json_schema` response_format type has a bug (Issue #10732). The error message
doesn't list it as valid, and schema extraction has issues.

### Solution
Use `json_object` type with nested `schema` field:

```python
# This format is supported in llama.cpp server-common.cpp:
payload["response_format"] = {
    "type": "json_object",
    "schema": your_json_schema  # Extracted and used for grammar!
}
```

From llama.cpp source (server-common.cpp):
```cpp
if (response_type == "json_object") {
    json_schema = json_value(response_format, "schema", json::object());  // Works!
} else if (response_type == "json_schema") {
    // Has issues - PR #18963 pending fix
}
```

### Verification
Response should conform to schema. Check logs for grammar compilation messages.

---

## Problem 4: All Outputs Use Function Call JSON Format

### Symptoms
- ALL LLM responses are JSON function calls: `{"name": "...", "parameters": {...}}`
- Even passes that should output plain text return JSON
- `forceJsonOutput` flag makes no difference - everything is JSON
- Model seems to be in "tool-calling mode" regardless of settings

### Root Cause
llama-server **auto-detects** the chat template from the model's metadata. Models like
Llama-3.1-8B-Instruct have a built-in **tool-calling template** that, when auto-detected,
forces ALL outputs into function call JSON format.

This happens even when:
- `forceJsonOutput` is false
- No `response_format` is specified
- No JSON schema is provided
- No tools are passed in the API request

### Solution
Create a **custom Jinja template** without tool-calling logic:

**Step 1:** Create `templates/llama31-no-tools.jinja`:
```jinja
{# Llama 3.1 chat template WITHOUT tool-calling logic #}
{{- bos_token }}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
{{- '<|start_header_id|>system<|end_header_id|>\n\n' }}
{{- message['content'] | trim }}
{{- '<|eot_id|>' }}
    {%- elif message['role'] == 'user' %}
{{- '<|start_header_id|>user<|end_header_id|>\n\n' }}
{{- message['content'] | trim }}
{{- '<|eot_id|>' }}
    {%- elif message['role'] == 'assistant' %}
{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- if message['content'] %}
{{- message['content'] | trim }}
{%- endif %}
{{- '<|eot_id|>' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
```

**Step 2:** Start llama-server with the custom template:
```bash
llama-server --jinja --chat-template-file /path/to/llama31-no-tools.jinja ...
```

### Why Built-in Templates Don't Work
- `--chat-template llama3` - Uses Llama 3.0 format, may cause system prompt issues with 3.1
- `--chat-template chatml` - Different token format, causes special token leakage
- No override - Auto-detects tool-calling template, forces JSON output

### Important Notes
- The custom template preserves proper Llama 3.1 message formatting
- `response_format: {"type": "json_object", "schema": {...}}` still works for JSON passes
- Template controls input formatting; response_format controls output grammar
- These are independent systems that work together correctly

### Verification
After restart:
- Passes WITHOUT `forceJsonOutput` should return plain text
- Passes WITH `forceJsonOutput` should return schema-constrained JSON
- No JSON function call format (`{"name": "...", "parameters": {...}}`) unless explicitly requested

---

## Quick Reference

| Issue | Symptom | Fix |
|-------|---------|-----|
| Slot 500 errors | `/slots` returns 500 | Add `--slots` flag |
| Stream stalls | No tokens for 60s | Use `json_object` + `schema` |
| JSON truncated | `finish_reason: "length"` | Increase `max_tokens` |
| Schema ignored | Any JSON returned | Use `json_object` not `json_schema` |
| All outputs JSON | Everything is function call format | Add `--chat-template llama3` |

## Notes

- These issues were discovered in llama.cpp b7662 but may persist in later versions
- PR #18963 (January 2026) fixes the `json_schema` type handling
- Grammar-based JSON constraints can have performance implications
- Always verify with latest llama.cpp documentation as fixes are merged

## References

- [llama.cpp Server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Issue #10732 - json_schema not working](https://github.com/ggml-org/llama.cpp/issues/10732)
- [PR #18963 - Fix json_schema handling](https://github.com/ggml-org/llama.cpp/pull/18963)
- [Discussion #6277 - Grammar blocking EOS](https://github.com/ggml-org/llama.cpp/discussions/6277)
- [KV Cache Slot Tutorial](https://github.com/ggml-org/llama.cpp/discussions/13606)
