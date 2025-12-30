"""NPC chat mode - talk to NPCs using the Loreguard API.

Uses the Loreguard API endpoints:
- GET  /api/engine/characters - List your registered NPCs
- POST /api/chat - Chat with an NPC (uses the multi-pass pipeline)
"""

import asyncio
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


# Loreguard API base URL
LOREGUARD_API_URL = "https://api.loreguard.com"


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
    """Client for the Loreguard API."""

    def __init__(self, api_token: str, base_url: str = LOREGUARD_API_URL):
        self.api_token = api_token
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    async def list_characters(self) -> list[NPCCharacter]:
        """Fetch list of registered NPCs."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/api/engine/characters",
                headers=self._headers(),
            )
            response.raise_for_status()
            data = response.json()

        return [NPCCharacter(id=c["id"], name=c["name"]) for c in data]

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

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            return response.json()


class NPCChat:
    """Interactive NPC chat session using Loreguard API."""

    def __init__(self, api_token: str, base_url: str = LOREGUARD_API_URL):
        self.client = LoreguardClient(api_token, base_url)
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
                raise Exception("Invalid API token - please check your authentication")
            elif e.response.status_code == 404:
                raise Exception("No characters found - register NPCs at loreguard.com first")
            raise
        except httpx.RequestError as e:
            raise Exception(f"Cannot connect to Loreguard API: {e}")

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
        print(f"{c(Colors.MUTED)}  Type your message and press Enter. Type 'quit' to exit.{c(Colors.RESET)}")
        print(f"{c(Colors.MUTED)}  Type '/debug' to show citations for the last response.{c(Colors.RESET)}")
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

            if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
                break

            # Debug command - show last response details
            if user_input.lower() == "/debug":
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


async def run_npc_chat(api_token: str, base_url: str = LOREGUARD_API_URL) -> None:
    """Run the NPC chat mode.

    Args:
        api_token: Loreguard API token for authentication
        base_url: Loreguard API base URL (default: https://api.loreguard.com)
    """
    chat = NPCChat(api_token=api_token, base_url=base_url)

    while True:
        npc = await chat.select_npc()
        if npc is None:
            break

        await chat.chat(npc)

        # Ask if they want to chat with another NPC
        print()
        continue_menu = Menu(
            items=[
                MenuItem(label="Chat with another NPC", value="continue"),
                MenuItem(label="Exit chat mode", value="exit"),
            ],
            title="Continue?",
            prompt="What would you like to do?",
        )
        choice = continue_menu.run()
        if choice is None or choice.value == "exit":
            break
