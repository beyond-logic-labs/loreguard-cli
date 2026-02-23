"""Claude CLI LLM adapter for testing with Haiku/Sonnet/Opus.

Calls `claude -p` (Claude Code CLI in non-interactive mode) as a subprocess,
implementing the same interface as LLMProxy. The engine stays completely
untouched — it sends the same requests, gets the same response format.

No streaming: generate_streaming() calls generate() and emits the full
response as a single token + done event (same pattern LLMProxy uses for
JSON mode).
"""

import asyncio
import json
import logging
import os
from dataclasses import field
from typing import Any, AsyncGenerator, Optional

from .config import load_config
from .llm import LLMRequest, LLMResponse, SamplingConfig, DEFAULT_STOP_SEQUENCES

logger = logging.getLogger(__name__)


class ClaudeLLMProxy:
    """
    Proxy to Claude via the `claude` CLI tool.

    Uses `claude -p` (non-interactive print mode) with --system-prompt
    and --output-format text. Each call is a fresh subprocess invocation.
    """

    def __init__(self, model: str = "haiku", timeout: float = 120.0):
        if not model:
            raise ValueError("Claude model is required")
        self.model = model
        self.default_timeout = timeout
        self.endpoint = "claude-cli"
        self.llm_type = "claude"

        config = load_config()
        self.max_message_length = config["MAX_MESSAGE_LENGTH"]
        self.max_total_context = config["MAX_TOTAL_CONTEXT"]
        self.max_timeout = config["MAX_TIMEOUT"]
        self.context_compaction = config["CONTEXT_COMPACTION"]

    async def close(self):
        """No-op — no persistent connections to close."""
        pass

    def _subprocess_env(self) -> dict:
        """Build env for claude subprocess — unset CLAUDECODE to allow nesting."""
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        return env

    async def check(self) -> bool:
        """Check if claude CLI is available."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "claude", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._subprocess_env(),
            )
            await asyncio.wait_for(proc.communicate(), timeout=10.0)
            return proc.returncode == 0
        except Exception:
            return False

    async def generate(self, request: dict | LLMRequest) -> dict:
        """
        Generate text via claude CLI.

        Same interface as LLMProxy.generate().
        """
        try:
            if isinstance(request, dict):
                req = self._dict_to_request(request)
            else:
                req = request

            req.messages = self._validate_messages(req.messages)
            return await self._call_claude(req)

        except ValueError as e:
            logger.warning(f"Validation error: {e}")
            return {"error": str(e), "content": ""}
        except asyncio.TimeoutError:
            logger.warning("Claude CLI request timed out")
            return {"error": "Claude CLI request timed out", "content": ""}
        except Exception as e:
            logger.exception("Claude CLI generation error")
            return {"error": "Internal Claude CLI error", "content": ""}

    async def generate_streaming(
        self, request: dict | LLMRequest
    ) -> AsyncGenerator[dict, None]:
        """
        Fake streaming — call generate(), emit as single token + done.

        Same pattern LLMProxy uses for JSON mode fallback.
        """
        result = await self.generate(request)

        if result.get("error"):
            yield {"type": "error", "error": result["error"]}
        else:
            content = result.get("content", "")
            yield {"type": "token", "token": content, "index": 0}
            yield {
                "type": "done",
                "content": content,
                "thinking": result.get("thinking", ""),
                "usage": result.get("usage", {}),
                "model": self.model,
                "token_count": 1,
                "finish_reason": "stop",
            }

    async def save_slot(self, slot_id: int, filename: str) -> dict:
        """No-op — Claude CLI doesn't have KV cache slots."""
        logger.debug(f"Slot save no-op (claude backend): slot={slot_id}, file={filename}")
        return {"n_saved": 0, "n_written": 0}

    async def restore_slot(self, slot_id: int, filename: str) -> dict:
        """No-op — return cold_start to signal no cache."""
        logger.debug(f"Slot restore no-op (claude backend): slot={slot_id}, file={filename}")
        return {"cold_start": True}

    async def _call_claude(self, req: LLMRequest) -> dict:
        """Execute claude -p and return the response."""
        # Extract system prompt from first message if it's a system message
        system_prompt = None
        conversation_messages = []

        for msg in req.messages:
            if msg.get("role") == "system" and system_prompt is None:
                system_prompt = msg["content"]
            else:
                conversation_messages.append(msg)

        # Format the user prompt
        prompt = self._format_prompt(conversation_messages)

        if not prompt:
            return {"error": "Empty prompt after formatting", "content": ""}

        # Build command — note: --tools must come LAST before the prompt
        # because it's variadic and an empty string can confuse arg parsing
        cmd = [
            "claude", "-p",
            "--model", self.model,
            "--output-format", "text",
        ]

        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        # JSON mode: include schema in prompt rather than --json-schema flag
        # to avoid output format interaction issues
        if req.force_json:
            if req.json_schema:
                schema_str = json.dumps(req.json_schema, indent=2)
                prompt += f"\n\nRespond with valid JSON matching this schema:\n{schema_str}\n\nOutput ONLY the JSON object. No markdown, no explanation, no code fences."
            else:
                prompt += "\n\nRespond with valid JSON only. No markdown, no explanation."

        # Max tokens via append-system-prompt (claude CLI doesn't have a direct flag)
        if req.max_tokens and req.max_tokens < 4096:
            cmd.extend([
                "--append-system-prompt",
                f"Keep your response under {req.max_tokens} tokens.",
            ])

        cmd.append(prompt)

        # Log request summary (cmd without the prompt for readability)
        total_chars = sum(len(m.get("content", "")) for m in req.messages)
        cmd_summary = " ".join(cmd[:-1])  # everything except the prompt
        logger.info(
            f"Claude CLI request: model={self.model}, {len(req.messages)} messages, "
            f"{total_chars} chars, max_tokens={req.max_tokens}, "
            f"force_json={req.force_json}, prompt_len={len(prompt)}"
        )
        logger.debug(f"Claude CLI cmd: {cmd_summary} <prompt:{len(prompt)} chars>")

        # Execute subprocess
        timeout = req.timeout or self.default_timeout
        timeout = min(timeout, self.max_timeout)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            env=self._subprocess_env(),
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()

        # Always log stderr for debugging (claude CLI may print warnings there)
        if stderr_text:
            logger.info(f"Claude CLI stderr: {stderr_text[:500]}")

        if proc.returncode != 0:
            error_msg = stderr_text[:500] if stderr_text else f"Exit code {proc.returncode}"
            logger.warning(f"Claude CLI failed (rc={proc.returncode}): {error_msg}")
            return {"error": f"Claude CLI error: {error_msg}", "content": ""}

        if not stdout_text:
            logger.warning(f"Claude CLI returned empty stdout (rc=0, stderr={stderr_text[:200]})")
            return {"error": "Empty response from Claude CLI", "content": ""}

        logger.info(f"Claude CLI response: {len(stdout_text)} chars")

        return {
            "content": stdout_text,
            "thinking": "",
            "model": self.model,
            "usage": {},
            "finish_reason": "stop",
        }

    def _format_prompt(self, messages: list[dict]) -> str:
        """Format conversation messages into a single prompt string.

        If there's only one user message, return it directly.
        If there's conversation history, format it with role labels.
        """
        if not messages:
            return ""

        # Single user message — pass directly
        if len(messages) == 1:
            return messages[0].get("content", "")

        # Multiple messages — format as conversation
        parts = []
        last_idx = len(messages) - 1

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if i == last_idx:
                # Last message is the actual request, separate it
                if parts:
                    parts.append("")  # blank line separator
                parts.append(content)
            else:
                # History messages get role labels
                label = "Player" if role == "user" else "NPC"
                parts.append(f"{label}: {content}")

        return "\n".join(parts)

    # =========================================================================
    # Shared utilities (same logic as LLMProxy, no httpx dependency)
    # =========================================================================

    def _validate_messages(self, messages: list[dict]) -> list[dict]:
        """Validate message structure and size limits."""
        if not messages:
            raise ValueError("Messages list is empty")

        for i, msg in enumerate(messages):
            if "role" not in msg:
                raise ValueError(f"Message {i} missing 'role'")
            if "content" not in msg:
                raise ValueError(f"Message {i} missing 'content'")
            if len(msg["content"]) > self.max_message_length:
                raise ValueError(
                    f"Message {i} too large: {len(msg['content'])} > {self.max_message_length}"
                )

        total_size = sum(len(m.get("content", "")) for m in messages)
        if total_size > self.max_total_context:
            if self.context_compaction:
                messages = self._compact_context(messages)
            else:
                raise ValueError(
                    f"Total context too large: {total_size} > {self.max_total_context}"
                )

        return messages

    def _compact_context(self, messages: list[dict]) -> list[dict]:
        """Compact context by removing older messages to fit within limits."""
        if not messages:
            return messages

        system_msg = None
        conversation = []
        for msg in messages:
            if msg.get("role") == "system" and system_msg is None:
                system_msg = msg
            else:
                conversation.append(msg)

        system_size = len(system_msg.get("content", "")) if system_msg else 0
        if system_size > self.max_total_context:
            raise ValueError(
                f"System message alone exceeds context limit: {system_size} > {self.max_total_context}"
            )

        available_space = self.max_total_context - system_size
        compacted = []
        current_size = 0

        for msg in reversed(conversation):
            msg_size = len(msg.get("content", ""))
            if current_size + msg_size <= available_space:
                compacted.insert(0, msg)
                current_size += msg_size
            else:
                break

        if not compacted and conversation:
            compacted = [conversation[-1]]
            logger.warning("Context compaction: keeping only most recent message")

        removed_count = len(conversation) - len(compacted)
        if removed_count > 0:
            logger.info(f"Context compaction: removed {removed_count} older messages")

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
            cache_prompt=d.get("cache_prompt", False),
        )
