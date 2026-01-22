"""LLM proxy for local llama.cpp server.

Supports llama-server's OpenAI-compatible /v1/chat/completions API with:
- Full sampling configuration (top_p, min_p, repeat_penalty, etc.)
- Stop sequences to prevent hallucinated conversation turns
- Thinking mode control (enable_thinking for Qwen3)
- JSON schema/response_format for structured output
- Thinking tag extraction (<think>...</think>)

Based on netshell's local_llm.go implementation.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import httpx

from .config import load_config
from .llama_server import get_slot_cache_dir

logger = logging.getLogger(__name__)


@dataclass
class SamplingConfig:
    """LLM sampling parameters for controlling generation behavior."""
    top_k: int = 0              # Limits vocabulary to top K tokens (0 = disabled)
    top_p: float = 0.9          # Nucleus sampling threshold (0-1, 1.0 = disabled)
    min_p: float = 0.05         # Minimum probability threshold (filters low-prob tokens)
    repeat_penalty: float = 1.05  # Penalty for repeating tokens (1.0 = no penalty)
    repeat_last_n: int = 2048   # How far back to look for repetitions
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


# Default stop sequences - ChatML/instruction markers that signal end of turn
DEFAULT_STOP_SEQUENCES = [
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "</s>",
    "<|end|>",
    "<|user|>",
    "<|assistant|>",
]


@dataclass
class LLMRequest:
    """Request for LLM generation."""
    messages: list[dict[str, str]]  # Chat messages
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: Optional[float] = None  # Per-request timeout (uses client default if None)
    sampling: Optional[SamplingConfig] = None
    stop: list[str] = field(default_factory=lambda: DEFAULT_STOP_SEQUENCES.copy())

    # Thinking mode control (for Qwen3)
    disable_thinking: bool = False

    # If true, error if content is empty instead of falling back to reasoning_content
    require_content: bool = False

    # JSON output mode
    force_json: bool = False
    json_schema: Optional[dict] = None

    # ADR-0014: KV cache prefix sharing for warmup context
    # When True, llama-server will cache the prompt prefix (system message)
    # and reuse it across requests with identical prefixes.
    cache_prompt: bool = False


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    content: str
    thinking: str = ""  # Extracted <think>...</think> content
    model: str = ""
    usage: dict = field(default_factory=dict)
    error: Optional[str] = None


class LLMProxy:
    """
    Proxy to local llama.cpp server.

    Uses the OpenAI-compatible /v1/chat/completions API with full
    sampling configuration, stop sequences, and JSON mode support.
    """

    def __init__(self, endpoint: str, timeout: float = 120.0):
        if not endpoint:
            raise ValueError("LLM endpoint is required")
        self.endpoint = endpoint.rstrip("/")
        self.default_timeout = timeout
        self.client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0,
            ),
        )
        self.llm_type = "llama.cpp"

        # Load limits from config
        config = load_config()
        self.max_message_length = config["MAX_MESSAGE_LENGTH"]
        self.max_total_context = config["MAX_TOTAL_CONTEXT"]
        self.max_timeout = config["MAX_TIMEOUT"]
        self.context_compaction = config["CONTEXT_COMPACTION"]

    async def close(self):
        """Close the HTTP client. Call on shutdown."""
        await self.client.aclose()

    async def check(self) -> bool:
        """Check if llama.cpp server is available."""
        try:
            response = await self.client.get(f"{self.endpoint}/v1/models")
            return response.status_code == 200
        except Exception:
            return False

    async def generate(self, request: dict | LLMRequest) -> dict:
        """
        Generate text from llama.cpp server.

        Request format (from backend):
        {
            "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
            "max_tokens": 512,
            "temperature": 0.7,
            "model": "default",
            "timeout": 120.0,  # optional, per-request timeout
            "sampling": {"top_p": 0.9, "min_p": 0.05, ...},  # optional
            "stop": ["<|im_end|>", ...],  # optional
            "disable_thinking": false,  # optional
            "require_content": false,  # optional, error if content empty
            "force_json": false,  # optional
            "json_schema": {...}  # optional
        }
        """
        try:
            # Convert dict to LLMRequest if needed
            if isinstance(request, dict):
                req = self._dict_to_request(request)
            else:
                req = request

            # Validate and possibly compact messages
            req.messages = self._validate_messages(req.messages)

            return await self._generate_llamacpp(req)
        except httpx.TimeoutException:
            logger.warning("LLM request timed out")
            return {"error": "LLM request timed out", "content": ""}
        except httpx.ConnectError:
            logger.warning(f"Could not connect to LLM server at {self.endpoint}")
            return {"error": "Could not connect to LLM server", "content": ""}
        except ValueError as e:
            # Validation errors - safe to return
            logger.warning(f"Validation error: {e}")
            return {"error": str(e), "content": ""}
        except Exception as e:
            # Log full error but return sanitized message
            logger.exception("LLM generation error")
            return {"error": "Internal LLM error", "content": ""}

    async def generate_streaming(
        self, request: dict | LLMRequest
    ) -> AsyncGenerator[dict, None]:
        """
        Generate text from llama.cpp server with token-by-token streaming.

        Yields chunks as they arrive:
        - {"type": "token", "token": "...", "index": N} for each token
        - {"type": "done", "content": "...", "thinking": "...", "usage": {...}} on completion
        - {"type": "error", "error": "..."} on error

        Request format is the same as generate().
        """
        try:
            # Convert dict to LLMRequest if needed
            if isinstance(request, dict):
                req = self._dict_to_request(request)
            else:
                req = request

            # Validate and possibly compact messages
            req.messages = self._validate_messages(req.messages)

            async for chunk in self._generate_llamacpp_streaming(req):
                yield chunk

        except httpx.TimeoutException:
            logger.warning("LLM streaming request timed out")
            yield {"type": "error", "error": "LLM request timed out"}
        except httpx.ConnectError:
            logger.warning(f"Could not connect to LLM server at {self.endpoint}")
            yield {"type": "error", "error": "Could not connect to LLM server"}
        except ValueError as e:
            logger.warning(f"Validation error: {e}")
            yield {"type": "error", "error": str(e)}
        except Exception as e:
            logger.exception("LLM streaming error")
            yield {"type": "error", "error": "Internal LLM error"}

    async def _generate_llamacpp_streaming(
        self, req: LLMRequest
    ) -> AsyncGenerator[dict, None]:
        """Generate using llama.cpp's OpenAI-compatible streaming API."""
        # Get sampling config (use defaults if not provided)
        sampling = req.sampling or SamplingConfig()

        # Cap max_tokens for safety
        max_tokens = min(req.max_tokens, 4096) if req.max_tokens > 0 else 512

        # Log request summary
        total_chars = sum(len(m.get("content", "")) for m in req.messages)
        logger.debug(
            f"LLM streaming request: {len(req.messages)} messages, {total_chars} chars, max_tokens={max_tokens}"
        )

        # Build request payload with stream=True
        payload: dict[str, Any] = {
            "messages": req.messages,
            "temperature": req.temperature,
            "max_tokens": max_tokens,
            "stream": True,  # Enable streaming
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
            "repeat_penalty": sampling.repeat_penalty,
            "repeat_last_n": sampling.repeat_last_n,
            "frequency_penalty": sampling.frequency_penalty,
            "presence_penalty": sampling.presence_penalty,
            "stop": req.stop,
        }

        # ADR-0014: Enable KV cache prefix sharing for warmup context
        if req.cache_prompt:
            payload["cache_prompt"] = True
            # Pin to slot 0 to ensure save/restore targets the correct slot.
            # With -np 1 (enforced when slot caching enabled), this is redundant
            # but kept for explicitness and defense-in-depth.
            # NOTE: llama-server must be started with -np 1 for VRAM efficiency;
            # id_slot alone does not prevent multi-slot allocation.
            payload["id_slot"] = 0
            logger.info("KV cache: cache_prompt=true, id_slot=0 (verify -np 1 on server)")

        # Disable thinking mode if requested (for Qwen3)
        if req.disable_thinking:
            payload["enable_thinking"] = False

        # Note: JSON mode is not compatible with streaming in llama.cpp
        # If force_json is requested, fall back to non-streaming
        if req.force_json:
            logger.debug(f"JSON MODE: force_json=True, schema={bool(req.json_schema)}")
            result = await self._generate_llamacpp(req)
            if result.get("error"):
                yield {"type": "error", "error": result["error"]}
            else:
                # Yield the complete response as a single "token"
                yield {
                    "type": "token",
                    "token": result.get("content", ""),
                    "index": 0,
                }
                yield {
                    "type": "done",
                    "content": result.get("content", ""),
                    "thinking": result.get("thinking", ""),
                    "usage": result.get("usage", {}),
                }
            return

        # Use per-request timeout if specified
        timeout = req.timeout or self.default_timeout

        # Track accumulated content for final processing
        accumulated_content = []
        token_index = 0
        usage = {}
        line_count = 0  # Track SSE lines for debugging

        try:
            # Use a custom timeout for streaming:
            # - connect: 10 seconds (fail fast if server unreachable)
            # - read: 60 seconds (max time between tokens - detects stalled stream)
            # - write: 30 seconds (sending the request)
            # - pool: 10 seconds (getting a connection from pool)
            #
            # Note: 60s between tokens allows for complex reasoning pauses
            # while still detecting genuinely stalled streams.
            stream_timeout = httpx.Timeout(
                connect=10.0,
                read=60.0,  # Max 60s between tokens (allows complex reasoning pauses)
                write=30.0,
                pool=10.0,
            )
            logger.info(f"STREAMING: Sending POST with stream=True, max_tokens={payload.get('max_tokens')}")
            async with self.client.stream(
                "POST",
                f"{self.endpoint}/v1/chat/completions",
                json=payload,
                timeout=stream_timeout,
            ) as response:
                logger.info(f"STREAMING: Got response status={response.status_code}")
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_text = error_text.decode()[:500]
                    logger.warning(f"llama.cpp streaming returned {response.status_code}: {error_text}")
                    yield {"type": "error", "error": f"LLM server error ({response.status_code})"}
                    return

                # Parse SSE stream
                logger.info("STREAMING: Starting to read SSE lines")
                async for line in response.aiter_lines():
                    line_count += 1
                    if line_count == 1:
                        logger.info(f"STREAMING: First line received: {line[:100] if line else '(empty)'}")
                    if not line:
                        continue

                    # SSE format: "data: {...}" or "data: [DONE]"
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix

                    # Check for end of stream
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        chunk_data = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse SSE chunk: {data_str[:100]}")
                        continue

                    # Extract token from delta
                    choices = chunk_data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "")

                    # Check for finish_reason
                    finish_reason = choices[0].get("finish_reason")

                    # Extract usage if present (some servers send it with final chunk)
                    if "usage" in chunk_data:
                        usage = chunk_data["usage"]

                    if token:
                        accumulated_content.append(token)
                        yield {
                            "type": "token",
                            "token": token,
                            "index": token_index,
                        }
                        token_index += 1

                    # Stop on finish_reason
                    if finish_reason:
                        break

        except httpx.TimeoutException as e:
            # Timeout between tokens (60s) - LLM likely hung or response not streaming
            partial_content = "".join(accumulated_content)
            logger.warning(f"STREAMING TIMEOUT: {token_index} tokens, {len(partial_content)} chars, lines_read={line_count}, exception={e}")
            yield {"type": "error", "error": f"LLM streaming timed out after {token_index} tokens (no token for 60s)"}
            return

        # Process final content
        raw_content = "".join(accumulated_content)

        if not raw_content:
            yield {"type": "error", "error": "Empty content from LLM"}
            return

        # Extract thinking from content
        thinking, content = self._extract_thinking(raw_content)

        # Strip chat markers
        content = self._strip_chat_markers(content)

        logger.debug(f"LLM streaming complete: {len(content)} chars, {token_index} tokens")

        yield {
            "type": "done",
            "content": content,
            "thinking": thinking,
            "usage": usage,
            "model": req.model,
            "token_count": token_index,
        }

    def _validate_messages(self, messages: list[dict]) -> list[dict]:
        """Validate message structure and size limits. Returns (possibly compacted) messages."""
        if not messages:
            raise ValueError("Messages list is empty")

        for i, msg in enumerate(messages):
            if "role" not in msg:
                raise ValueError(f"Message {i} missing 'role'")
            if "content" not in msg:
                raise ValueError(f"Message {i} missing 'content'")

            content = msg["content"]
            if len(content) > self.max_message_length:
                raise ValueError(f"Message {i} too large: {len(content)} > {self.max_message_length}")

        total_size = sum(len(m.get("content", "")) for m in messages)

        if total_size > self.max_total_context:
            if self.context_compaction:
                messages = self._compact_context(messages)
            else:
                raise ValueError(f"Total context too large: {total_size} > {self.max_total_context}")

        return messages

    def _compact_context(self, messages: list[dict]) -> list[dict]:
        """
        Compact context by removing older messages to fit within limits.

        Strategy:
        1. Always keep the system message (first message if role=system)
        2. Always keep the most recent user message
        3. Remove oldest non-system messages until under limit
        """
        if not messages:
            return messages

        # Separate system message from conversation
        system_msg = None
        conversation = []

        for msg in messages:
            if msg.get("role") == "system" and system_msg is None:
                system_msg = msg
            else:
                conversation.append(msg)

        # Calculate base size (system message)
        system_size = len(system_msg.get("content", "")) if system_msg else 0

        # If system message alone exceeds limit, we can't compact
        if system_size > self.max_total_context:
            raise ValueError(f"System message alone exceeds context limit: {system_size} > {self.max_total_context}")

        available_space = self.max_total_context - system_size

        # Build compacted conversation from most recent messages
        compacted = []
        current_size = 0

        # Process from newest to oldest
        for msg in reversed(conversation):
            msg_size = len(msg.get("content", ""))
            if current_size + msg_size <= available_space:
                compacted.insert(0, msg)
                current_size += msg_size
            else:
                # Stop adding older messages
                break

        # Ensure we kept at least the most recent message
        if not compacted and conversation:
            # Force include the last message even if it's too big (will be validated later)
            compacted = [conversation[-1]]
            logger.warning("Context compaction: keeping only most recent message")

        # Log compaction info
        removed_count = len(conversation) - len(compacted)
        if removed_count > 0:
            logger.info(f"Context compaction: removed {removed_count} older messages")

        # Reconstruct messages with system prompt first
        result = []
        if system_msg:
            result.append(system_msg)
        result.extend(compacted)

        return result

    def _dict_to_request(self, d: dict) -> LLMRequest:
        """Convert a dict request to LLMRequest."""
        sampling = None
        if "sampling" in d and d["sampling"]:
            s = d["sampling"]
            sampling = SamplingConfig(
                top_k=s.get("top_k", 0),
                top_p=s.get("top_p", 0.9),
                min_p=s.get("min_p", 0.05),
                repeat_penalty=s.get("repeat_penalty", 1.05),
                repeat_last_n=s.get("repeat_last_n", 2048),
                frequency_penalty=s.get("frequency_penalty", 0.0),
                presence_penalty=s.get("presence_penalty", 0.0),
            )

        # Validate and cap timeout
        timeout = d.get("timeout")
        if timeout is not None:
            timeout = min(float(timeout), self.max_timeout)

        return LLMRequest(
            messages=d.get("messages", []),
            model=d.get("model", "default"),
            temperature=d.get("temperature", 0.7),
            max_tokens=d.get("max_tokens", 512),
            timeout=timeout,
            sampling=sampling,
            stop=d.get("stop", DEFAULT_STOP_SEQUENCES.copy()),
            disable_thinking=d.get("disable_thinking", False),
            require_content=d.get("require_content", False),
            force_json=d.get("force_json", False),
            json_schema=d.get("json_schema"),
            cache_prompt=d.get("cache_prompt", False),  # ADR-0014: KV cache prefix sharing
        )

    async def _generate_llamacpp(self, req: LLMRequest) -> dict:
        """Generate using llama.cpp's OpenAI-compatible API."""
        # Get sampling config (use defaults if not provided)
        sampling = req.sampling or SamplingConfig()

        # Cap max_tokens for safety
        max_tokens = min(req.max_tokens, 4096) if req.max_tokens > 0 else 512

        # Log request summary
        total_chars = sum(len(m.get("content", "")) for m in req.messages)
        logger.debug(f"LLM request: {len(req.messages)} messages, {total_chars} chars, max_tokens={max_tokens}")

        # Build request payload
        payload: dict[str, Any] = {
            "messages": req.messages,
            "temperature": req.temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
            "repeat_penalty": sampling.repeat_penalty,
            "repeat_last_n": sampling.repeat_last_n,
            "frequency_penalty": sampling.frequency_penalty,
            "presence_penalty": sampling.presence_penalty,
            "stop": req.stop,
        }

        # ADR-0014: Enable KV cache prefix sharing for warmup context
        if req.cache_prompt:
            payload["cache_prompt"] = True
            # Pin to slot 0 to ensure save/restore targets the correct slot.
            # With -np 1 (enforced when slot caching enabled), this is redundant
            # but kept for explicitness and defense-in-depth.
            # NOTE: llama-server must be started with -np 1 for VRAM efficiency;
            # id_slot alone does not prevent multi-slot allocation.
            payload["id_slot"] = 0
            logger.info("KV cache: cache_prompt=true, id_slot=0 (verify -np 1 on server)")

        # Disable thinking mode if requested (for Qwen3)
        if req.disable_thinking:
            payload["enable_thinking"] = False

        # Force JSON output if requested
        # Use json_object type with schema field - this is supported in llama.cpp server
        # (server-common.cpp extracts schema from response_format.schema for json_object type)
        # Note: json_schema type has a bug (issue #10732, PR #18963 pending)
        if req.force_json:
            # Merge system prompt into user message for "Content-only" template
            if len(req.messages) >= 2 and req.messages[0].get("role") == "system":
                system_content = req.messages[0]["content"]
                user_content = req.messages[-1]["content"]
                merged = f"INSTRUCTIONS:\n{system_content}\n\nREQUEST:\n{user_content}"
                payload["messages"] = [{"role": "user", "content": merged}]
                logger.debug("JSON MODE: Merged system into user message")

            # Use json_object with schema for proper constraint enforcement
            if req.json_schema:
                payload["response_format"] = {"type": "json_object", "schema": req.json_schema}
                logger.info(f"JSON MODE: response_format=json_object with schema")
            else:
                payload["response_format"] = {"type": "json_object"}
                logger.info(f"JSON MODE: response_format=json_object (no schema)")

        # Use per-request timeout if specified
        timeout = req.timeout or self.default_timeout

        response = await self.client.post(
            f"{self.endpoint}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )

        if response.status_code != 200:
            error_text = response.text[:500]  # Truncate error for safety
            logger.warning(f"llama.cpp returned {response.status_code}: {error_text}")
            return {"error": f"LLM server error ({response.status_code})", "content": ""}

        data = response.json()

        if not data.get("choices"):
            return {"error": "Empty response from LLM", "content": ""}

        msg = data["choices"][0].get("message", {})
        raw_content = msg.get("content", "")

        # Some models use reasoning_content for thinking
        if not raw_content and msg.get("reasoning_content"):
            if req.require_content:
                # Caller requires actual content, not reasoning - don't fall back
                logger.warning("Model produced reasoning but no final response")
                return {"error": "Model produced reasoning but no response (try increasing max_tokens)", "content": ""}
            logger.debug("Using reasoning_content as content")
            raw_content = msg["reasoning_content"]

        if not raw_content:
            return {"error": "Empty content from LLM", "content": ""}

        # Log response summary
        logger.debug(f"LLM response: {len(raw_content)} chars")

        # Extract thinking from <think>...</think> tags
        thinking, content = self._extract_thinking(raw_content)

        # Strip chat markers that might indicate hallucinated turns
        content = self._strip_chat_markers(content)

        return {
            "content": content,
            "thinking": thinking,
            "model": data.get("model", req.model),
            "usage": data.get("usage", {}),
        }

    def _extract_thinking(self, content: str) -> tuple[str, str]:
        """Extract <think>...</think> blocks from LLM response.

        Handles malformed tags like 'think>' (missing '<') that some models produce.
        Returns (thinking, clean_content) where clean_content has thinking removed.
        """
        # Try to find thinking markers (standard and malformed)
        start_idx = self._find_think_start(content)
        end_idx = self._find_think_end(content)

        if start_idx == -1:
            return "", content.strip()

        # Get content before <think> tag (valid response content)
        before_think = content[:start_idx].strip()

        # Calculate the length of the start tag we found
        if content[start_idx:start_idx+7].lower() == "<think>":
            start_tag_len = 7
        else:
            start_tag_len = 6  # "think>" without <

        # Malformed - has start but no end
        if end_idx == -1 or end_idx < start_idx:
            thinking = content[start_idx + start_tag_len:].strip()
            return thinking, before_think if before_think else ""

        # Extract thinking content (between tags)
        thinking = content[start_idx + start_tag_len:end_idx].strip()
        after_think = content[end_idx + 8:].strip()  # +8 for "</think>"
        clean_content = f"{before_think} {after_think}".strip()

        # Handle empty thinking (model sometimes outputs <think>\n\n</think>)
        if not thinking:
            return "", clean_content

        return thinking, clean_content

    def _find_think_start(self, content: str) -> int:
        """Find start of thinking block, handling malformed tags."""
        # Try standard tag first
        idx = content.lower().find("<think>")
        if idx != -1:
            return idx

        # Try malformed variant (missing <)
        idx = content.lower().find("think>")
        if idx != -1:
            # Make sure this isn't "</think>"
            if idx == 0 or content[idx - 1] != "/":
                return idx

        return -1

    def _find_think_end(self, content: str) -> int:
        """Find end of thinking block, handling malformed tags."""
        # Try standard tag first
        idx = content.lower().find("</think>")
        if idx != -1:
            return idx

        # Try malformed variant
        idx = content.lower().find("/think>")
        if idx != -1:
            return idx - 1  # Adjust to include the "<" position

        return -1

    def _strip_chat_markers(self, content: str) -> str:
        """Remove content after ChatML markers that indicate hallucinated turns."""
        markers = [
            "<|im_end|>", "<|im_start|>", "<|endoftext|>",
            "</s>", "<|end|>", "<|user|>", "<|assistant|>",
        ]

        result = content
        for marker in markers:
            if marker in result:
                idx = result.index(marker)
                result = result[:idx]

        return result.strip()

    async def list_models(self) -> list[str]:
        """List available models from llama.cpp server."""
        try:
            response = await self.client.get(f"{self.endpoint}/v1/models")
            if response.status_code == 200:
                data = response.json()
                return [m["id"] for m in data.get("data", [])]
        except Exception:
            pass
        return []

    # =========================================================================
    # ADR-0014: Slot KV Cache Persistence
    # =========================================================================

    def _validate_cache_filename(self, filename: str) -> str:
        """Validate and sanitize cache filename to prevent path traversal.

        Security: Ensures filename is safe for filesystem operations.
        Rejects any attempts at path traversal or injection.

        Returns:
            Sanitized filename

        Raises:
            ValueError: If filename is invalid or contains unsafe characters
        """
        import os
        import re
        from urllib.parse import quote

        if not filename:
            raise ValueError("Empty filename")

        # Extract basename to prevent directory traversal
        safe_name = os.path.basename(filename)

        # Reject if basename differs (means path separators were present)
        if safe_name != filename:
            raise ValueError(f"Path traversal detected in filename: {filename}")

        # Reject hidden files
        if safe_name.startswith("."):
            raise ValueError(f"Hidden files not allowed: {filename}")

        # Validate format: only allow safe characters
        if not re.match(r'^[a-zA-Z0-9_-]+\.bin$', safe_name):
            raise ValueError(f"Invalid filename format: {filename}")

        # URL-encode for safe transmission
        return quote(safe_name, safe='')

    async def save_slot(self, slot_id: int, filename: str) -> dict:
        """Save a slot's KV cache to disk.

        ADR-0014: This enables cross-message caching by persisting the warmup
        context KV state. Call after Pass 4 (non-JSON) to preserve the cache
        before Pass 5 (JSON) overwrites it.

        Args:
            slot_id: The slot to save (usually 0)
            filename: Cache filename (e.g., "npc_{id}_{hash}.bin")

        Returns:
            Dict with n_saved (tokens), n_written (bytes), timings, or error
        """
        try:
            # Validate and URL-encode filename
            safe_filename = self._validate_cache_filename(filename)

            response = await self.client.post(
                f"{self.endpoint}/slots/{slot_id}?action=save&filename={safe_filename}",
                json={},  # llama-server expects JSON body
                timeout=30.0,
            )
            if response.status_code == 200:
                data = response.json()
                logger.debug(
                    f"Slot {slot_id} saved: {data.get('n_saved', 0)} tokens, "
                    f"{data.get('n_written', 0)} bytes"
                )
                # ADR-0014: Clean up stale cache files for this NPC
                # Run in background to not block the response
                self._cleanup_stale_cache_files(filename)
                return data
            else:
                error = response.text[:200]
                logger.warning(f"Failed to save slot {slot_id}: {response.status_code} - {error}")
                return {"error": f"Save failed: {response.status_code}"}
        except Exception as e:
            logger.warning(f"Exception saving slot {slot_id}: {e}")
            return {"error": str(e)}

    def _cleanup_stale_cache_files(self, current_filename: str) -> None:
        """Remove stale cache files for ALL NPCs, keeping only the most recent per NPC.

        ADR-0014: After saving, clean up old cache files to prevent disk accumulation.
        For each NPC, keep only the newest cache file (by modification time).

        Filename format: npc_{npc_id}_{hash}.bin
        """
        try:
            cache_dir = get_slot_cache_dir()
            all_cache_files = list(cache_dir.glob("npc_*.bin"))

            if not all_cache_files:
                return

            # Group files by NPC prefix (everything before the last underscore)
            # Example: npc_zero_a1b2c3d4.bin -> npc_zero
            npc_files: dict[str, list[Path]] = {}
            for cache_file in all_cache_files:
                parts = cache_file.name.rsplit('_', 1)
                if len(parts) == 2 and parts[0].startswith('npc_'):
                    npc_prefix = parts[0]
                    if npc_prefix not in npc_files:
                        npc_files[npc_prefix] = []
                    npc_files[npc_prefix].append(cache_file)

            # For each NPC, keep only the newest file
            removed_count = 0
            for npc_prefix, files in npc_files.items():
                if len(files) <= 1:
                    continue  # Nothing to clean up

                # Sort by modification time, newest first
                files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

                # Delete all but the newest (files[0])
                for stale_file in files[1:]:
                    try:
                        stale_file.unlink()
                        removed_count += 1
                        logger.debug(f"Removed stale cache: {stale_file.name}")
                    except OSError as e:
                        logger.warning(f"Failed to remove {stale_file.name}: {e}")

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} stale cache file(s)")

        except Exception as e:
            # Don't fail the save operation if cleanup fails
            logger.warning(f"Cache cleanup error (non-fatal): {e}")

    async def restore_slot(self, slot_id: int, filename: str) -> dict:
        """Restore a slot's KV cache from disk.

        ADR-0014: This enables cross-message caching by restoring the warmup
        context KV state. Call before Pass 2 to warm the cache from a previous
        conversation turn.

        Args:
            slot_id: The slot to restore to (usually 0)
            filename: Cache filename (e.g., "npc_{id}_{hash}.bin")

        Returns:
            Dict with n_restored (tokens), n_read (bytes), timings, or error
        """
        try:
            # Validate and URL-encode filename
            safe_filename = self._validate_cache_filename(filename)

            response = await self.client.post(
                f"{self.endpoint}/slots/{slot_id}?action=restore&filename={safe_filename}",
                json={},  # llama-server expects JSON body
                timeout=30.0,
            )
            if response.status_code == 200:
                data = response.json()
                logger.debug(
                    f"Slot {slot_id} restored: {data.get('n_restored', 0)} tokens, "
                    f"{data.get('n_read', 0)} bytes"
                )
                return data
            elif response.status_code == 404:
                # Cache file doesn't exist - not an error, just cold start
                logger.debug(f"No cache file for slot {slot_id}: {filename}")
                return {"cold_start": True}
            else:
                error = response.text[:200]
                logger.warning(f"Failed to restore slot {slot_id}: {response.status_code} - {error}")
                return {"error": f"Restore failed: {response.status_code}"}
        except Exception as e:
            logger.warning(f"Exception restoring slot {slot_id}: {e}")
            return {"error": str(e)}
