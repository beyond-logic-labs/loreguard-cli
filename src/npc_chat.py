"""NPC chat mode - talk to NPCs using the Loreguard API.

Uses the Loreguard API endpoints:
- GET  /api/characters - List your registered NPCs
- POST /api/chat - Chat with an NPC (uses the multi-pass pipeline)

Supports two authentication modes:
- API Token: For server-side or development use (api_token parameter)
- Player JWT: For game clients using Steam authentication (player_jwt parameter)

Rate Limits (when using Player JWT):
- 60 requests/minute per player
- 1000 requests/minute per studio
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from .term_ui import (
    Colors,
    Menu,
    MenuItem,
    InputField,
    supports_color,
    print_info,
    print_error,
    print_success,
)


# Configure module logger
logger = logging.getLogger(__name__)

# Loreguard API base URL
LOREGUARD_API_URL = "https://api.loreguard.com"


@dataclass
class ClientConfig:
    """Configuration for Loreguard client.

    Attributes:
        connect_timeout: Timeout for establishing connection (seconds).
        read_timeout: Timeout for reading response (seconds).
        chat_timeout: Extended timeout for chat requests (seconds).
    """

    connect_timeout: float = 5.0
    read_timeout: float = 30.0
    chat_timeout: float = 120.0


class RateLimitError(Exception):
    """Rate limit exceeded error with details."""

    def __init__(
        self,
        message: str,
        limit_type: str = "unknown",
        requests_used: int = 0,
        requests_limit: int = 0,
        reset_at: Optional[str] = None,
    ):
        super().__init__(message)
        self.limit_type = limit_type
        self.requests_used = requests_used
        self.requests_limit = requests_limit
        self.reset_at = reset_at

    def __str__(self):
        if self.reset_at:
            return f"{self.args[0]} ({self.requests_used}/{self.requests_limit}, resets at {self.reset_at})"
        return f"{self.args[0]} ({self.requests_used}/{self.requests_limit})"


@dataclass
class NPCCharacter:
    """An NPC character from the Loreguard API."""
    id: str
    name: str


@dataclass
class ChatMessage:
    """A message in the conversation."""
    role: str  # "user" or "assistant"
    content: str


class LoreguardClient:
    """Client for the Loreguard API.

    Supports two authentication modes:
    - api_token: For server-side or development use
    - player_jwt: For game clients using Steam authentication

    Only one of api_token or player_jwt should be provided.
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        player_jwt: Optional[str] = None,
        base_url: str = LOREGUARD_API_URL,
        config: Optional[ClientConfig] = None,
    ):
        if not api_token and not player_jwt:
            raise ValueError("Either api_token or player_jwt must be provided")
        if api_token and player_jwt:
            raise ValueError("Only one of api_token or player_jwt should be provided")

        self.api_token = api_token
        self.player_jwt = player_jwt
        self.base_url = base_url.rstrip("/")
        self.config = config or ClientConfig()

    def _headers(self) -> dict:
        token = self.api_token or self.player_jwt
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _get_timeout(self, for_chat: bool = False) -> httpx.Timeout:
        """Get configured timeout settings.

        Args:
            for_chat: Use extended timeout for chat requests.
        """
        read_timeout = self.config.chat_timeout if for_chat else self.config.read_timeout
        return httpx.Timeout(
            connect=self.config.connect_timeout,
            read=read_timeout,
            write=read_timeout,
            pool=self.config.connect_timeout,
        )

    def _handle_rate_limit(self, response: httpx.Response) -> None:
        """Check for rate limit errors and raise RateLimitError with details."""
        if response.status_code == 429:
            try:
                data = response.json()
            except Exception:
                data = {}

            logger.warning(
                "Rate limited: %s/%s requests (type=%s, resets=%s)",
                data.get("requests_used", "?"),
                data.get("requests_limit", "?"),
                data.get("limit_type", "unknown"),
                data.get("reset_at", "?"),
            )

            raise RateLimitError(
                message=data.get("error", "Rate limit exceeded"),
                limit_type=data.get("limit_type", "unknown"),
                requests_used=data.get("requests_used", 0),
                requests_limit=data.get("requests_limit", 0),
                reset_at=data.get("reset_at"),
            )

    async def list_characters(self) -> list[NPCCharacter]:
        """Fetch list of registered NPCs."""
        logger.debug("Fetching character list from %s", self.base_url)

        async with httpx.AsyncClient(timeout=self._get_timeout()) as client:
            response = await client.get(
                f"{self.base_url}/api/characters",
                headers=self._headers(),
            )
            self._handle_rate_limit(response)
            response.raise_for_status()
            data = response.json()

        characters = [NPCCharacter(id=c["id"], name=c["name"]) for c in data]
        logger.info("Fetched %d characters", len(characters))
        return characters

    async def chat(
        self,
        character_id: str,
        message: str,
        history: list[ChatMessage] = None,
        player_id: str = "cli_user",
        context: str = "",
    ) -> dict:
        """Send a chat message to an NPC and get response.

        Returns the full API response with speech, thoughts, citations, etc.
        """
        logger.debug(
            "Chat request: character=%s, message_len=%d, history_len=%d",
            character_id,
            len(message),
            len(history) if history else 0,
        )

        history_data = []
        if history:
            history_data = [{"role": m.role, "content": m.content} for m in history]

        payload = {
            "character_id": character_id,
            "message": message,
            "history": history_data,
            "player_id": player_id,
        }
        if context:
            payload["current_context"] = context

        async with httpx.AsyncClient(timeout=self._get_timeout(for_chat=True)) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                headers=self._headers(),
                json=payload,
            )
            self._handle_rate_limit(response)
            response.raise_for_status()
            result = response.json()

        logger.info(
            "Chat response: character=%s, verified=%s, latency=%sms",
            character_id,
            result.get("verified", "N/A"),
            result.get("latency_ms", "N/A"),
        )
        return result


class NPCChat:
    """Interactive NPC chat session using Loreguard API.

    Supports two authentication modes:
    - api_token: For server-side or development use
    - player_jwt: For game clients using Steam authentication
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        player_jwt: Optional[str] = None,
        base_url: str = LOREGUARD_API_URL,
        config: Optional[ClientConfig] = None,
    ):
        self.client = LoreguardClient(
            api_token=api_token,
            player_jwt=player_jwt,
            base_url=base_url,
            config=config,
        )
        self.current_npc: Optional[NPCCharacter] = None
        self.history: list[ChatMessage] = []
        self.use_color = supports_color()

    def _c(self, color: str) -> str:
        return color if self.use_color else ""

    async def fetch_characters(self) -> list[NPCCharacter]:
        """Fetch available NPCs from the API."""
        try:
            return await self.client.list_characters()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error("Authentication failed: invalid API token")
                raise Exception("Invalid API token - please check your authentication")
            elif e.response.status_code == 404:
                logger.warning("No characters found for this account")
                raise Exception("No characters found - register NPCs at loreguard.com first")
            logger.error("HTTP error fetching characters: %d", e.response.status_code)
            raise
        except httpx.RequestError as e:
            logger.error("Connection error: %s", e)
            raise Exception("Cannot connect to Loreguard API. Check your network connection.")

    async def select_npc(self) -> Optional[NPCCharacter]:
        """Show NPC selection menu."""
        c = self._c

        print()
        print(f"{c(Colors.MUTED)}Fetching your NPCs from Loreguard...{c(Colors.RESET)}")

        try:
            characters = await self.fetch_characters()
        except Exception as e:
            print_error(str(e))
            return None

        if not characters:
            print_error("No NPCs registered. Create NPCs at loreguard.com first.")
            return None

        items = [
            MenuItem(
                label=npc.name,
                value=npc.id,
                description=f"ID: {npc.id}",
            )
            for npc in characters
        ]

        menu = Menu(
            items=items,
            title="Select NPC",
            prompt="Who do you want to talk to?",
        )

        selected = menu.run()
        if selected is None:
            return None

        for npc in characters:
            if npc.id == selected.value:
                return npc
        return None

    async def chat(self, npc: NPCCharacter) -> None:
        """Run the chat loop with an NPC."""
        self.current_npc = npc
        self.history = []
        c = self._c

        print()
        print(f"{c(Colors.CYAN)}{'─' * 60}{c(Colors.RESET)}")
        print(f"{c(Colors.BRIGHT_MAGENTA)}  Chatting with {npc.name}{c(Colors.RESET)}")
        print(f"{c(Colors.MUTED)}  Using Loreguard API (grounded responses){c(Colors.RESET)}")
        print(f"{c(Colors.CYAN)}{'─' * 60}{c(Colors.RESET)}")
        print(f"{c(Colors.MUTED)}  Commands: /help /switch /list /debug /quit{c(Colors.RESET)}")
        print()

        last_response: Optional[dict] = None

        while True:
            # Get user input
            try:
                user_input = input(f"{c(Colors.BRIGHT_GREEN)}You:{c(Colors.RESET)} ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().strip()

                if cmd in ("/quit", "/exit", "/back"):
                    return "back"

                if cmd == "/help":
                    print()
                    print(f"{c(Colors.MUTED)}Available commands:{c(Colors.RESET)}")
                    print(f"  {c(Colors.CYAN)}/switch{c(Colors.RESET)}  - Switch to another NPC")
                    print(f"  {c(Colors.CYAN)}/list{c(Colors.RESET)}    - List all available NPCs")
                    print(f"  {c(Colors.CYAN)}/debug{c(Colors.RESET)}   - Show details of last response")
                    print(f"  {c(Colors.CYAN)}/clear{c(Colors.RESET)}   - Clear conversation history")
                    print(f"  {c(Colors.CYAN)}/quit{c(Colors.RESET)}    - Exit chat")
                    print()
                    continue

                if cmd == "/switch":
                    return "switch"

                if cmd == "/list":
                    print()
                    print(f"{c(Colors.MUTED)}Fetching NPCs...{c(Colors.RESET)}")
                    try:
                        characters = await self.fetch_characters()
                        print(f"{c(Colors.MUTED)}Available NPCs:{c(Colors.RESET)}")
                        for char in characters:
                            current = " (current)" if char.id == npc.id else ""
                            print(f"  {c(Colors.CYAN)}•{c(Colors.RESET)} {char.name}{c(Colors.MUTED)}{current}{c(Colors.RESET)}")
                        print()
                    except Exception as e:
                        print_error(str(e))
                    continue

                if cmd == "/clear":
                    self.history = []
                    print(f"{c(Colors.MUTED)}Conversation history cleared.{c(Colors.RESET)}")
                    print()
                    continue

                if cmd == "/debug":
                    if last_response:
                        print()
                        print(f"{c(Colors.MUTED)}Last response details:{c(Colors.RESET)}")
                        print(f"  {c(Colors.LABEL)}Verified:{c(Colors.RESET)} {last_response.get('verified', 'N/A')}")
                        print(f"  {c(Colors.LABEL)}Latency:{c(Colors.RESET)} {last_response.get('latency_ms', 'N/A')}ms")
                        print(f"  {c(Colors.LABEL)}Retries:{c(Colors.RESET)} {last_response.get('retries', 0)}")
                        if last_response.get('thoughts'):
                            print(f"  {c(Colors.LABEL)}Thoughts:{c(Colors.RESET)} {last_response['thoughts']}")
                        citations = last_response.get('citations', [])
                        if citations:
                            print(f"  {c(Colors.LABEL)}Citations:{c(Colors.RESET)}")
                            for cit in citations:
                                verified = "✓" if cit.get('verified') else "✗"
                                print(f"    {verified} [{cit.get('evidence_id', '?')}] {cit.get('claim', '')[:50]}...")
                        print()
                    else:
                        print(f"{c(Colors.MUTED)}No previous response.{c(Colors.RESET)}")
                    continue

                # Unknown command
                print(f"{c(Colors.MUTED)}Unknown command. Type /help for available commands.{c(Colors.RESET)}")
                continue

            if user_input.lower() in ("quit", "exit"):
                return "back"

            # Show thinking indicator
            print(f"{c(Colors.MUTED)}  {npc.name} is thinking...{c(Colors.RESET)}", end="", flush=True)

            # Get response from Loreguard API
            try:
                response = await self.client.chat(
                    character_id=npc.id,
                    message=user_input,
                    history=self.history,
                )
                last_response = response

                # Clear thinking indicator
                print(f"\r{' ' * 50}\r", end="")

                # Extract speech
                speech = response.get("speech", "...")

                # Show response with verification indicator
                verified = response.get("verified", False)
                indicator = f"{c(Colors.BRIGHT_GREEN)}●{c(Colors.RESET)}" if verified else f"{c(Colors.YELLOW)}○{c(Colors.RESET)}"
                print(f"{c(Colors.BRIGHT_CYAN)}{npc.name}:{c(Colors.RESET)} {speech} {indicator}")
                print()

                # Update history
                self.history.append(ChatMessage(role="user", content=user_input))
                self.history.append(ChatMessage(role="assistant", content=speech))

                # Keep history manageable
                if len(self.history) > 20:
                    self.history = self.history[-20:]

            except RateLimitError as e:
                print(f"\r{' ' * 50}\r", end="")
                print_error(f"Rate limited: {e.requests_used}/{e.requests_limit} requests")
                if e.reset_at:
                    print_error(f"  Resets at: {e.reset_at}")
                print_info("  Wait a moment before sending more messages.")
            except httpx.HTTPStatusError as e:
                print(f"\r{' ' * 50}\r", end="")
                if e.response.status_code == 404:
                    print_error(f"NPC '{npc.id}' not found")
                elif e.response.status_code == 401:
                    print_error("Authentication failed - token may have expired")
                    break
                else:
                    print_error(f"API error: {e.response.status_code}")
            except httpx.RequestError as e:
                print(f"\r{' ' * 50}\r", end="")
                print_error(f"Connection error: {e}")
            except Exception as e:
                print(f"\r{' ' * 50}\r", end="")
                print_error(f"Error: {e}")

        print()
        print_info(f"Chat with {npc.name} ended.")
        return "back"


async def run_npc_chat(
    api_token: Optional[str] = None,
    player_jwt: Optional[str] = None,
    base_url: str = LOREGUARD_API_URL,
    config: Optional[ClientConfig] = None,
) -> None:
    """Run the NPC chat mode.

    Args:
        api_token: Loreguard API token for authentication (for server-side use)
        player_jwt: Player JWT from Steam exchange (for game clients)
        base_url: Loreguard API base URL (default: https://api.loreguard.com)
        config: Optional client configuration for timeouts

    Note:
        Only one of api_token or player_jwt should be provided.
        Player JWTs are subject to rate limiting (60 req/min per player).
    """
    chat = NPCChat(api_token=api_token, player_jwt=player_jwt, base_url=base_url, config=config)

    while True:
        npc = await chat.select_npc()
        if npc is None:
            break

        result = await chat.chat(npc)

        # Handle chat result
        if result == "switch":
            # Go back to NPC selection
            continue
        else:
            # "back" or any other - exit chat mode
            break
