"""Steam authentication for Loreguard API.

Exchange Steam session tickets for Player JWTs to enable player-authenticated NPC chat.
This is useful for game clients that want to authenticate players via Steam.

Usage:
    from src.steam import SteamAuth, SteamAuthConfig

    # With custom config
    config = SteamAuthConfig(
        connect_timeout=5.0,
        read_timeout=15.0,
        max_retries=3
    )
    steam_auth = SteamAuth(api_url="https://api.loreguard.com", config=config)

    # Exchange Steam ticket for Player JWT
    result = await steam_auth.exchange_ticket(
        app_id="480",  # Your Steam AppID
        ticket="base64-encoded-steam-ticket"
    )

    # Use the token for NPC chat
    player_jwt = result.token
    expires_at = result.expires_at
"""

import asyncio
import base64
import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx


# Configure module logger
logger = logging.getLogger(__name__)

# Default Loreguard API URL
LOREGUARD_API_URL = "https://api.loreguard.com"

# Validation patterns
STEAM_APP_ID_PATTERN = re.compile(r"^\d{1,10}$")


class SteamAuthError(Exception):
    """Error during Steam authentication."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class RateLimitError(SteamAuthError):
    """Rate limit exceeded during Steam authentication."""

    def __init__(
        self,
        message: str,
        limit_type: str,
        requests_used: int,
        requests_limit: int,
        reset_at: Optional[str] = None,
    ):
        super().__init__(message, status_code=429, error_code="rate_limited")
        self.limit_type = limit_type
        self.requests_used = requests_used
        self.requests_limit = requests_limit
        self.reset_at = reset_at


class ValidationError(SteamAuthError):
    """Input validation error."""

    def __init__(self, message: str, field: str):
        super().__init__(message, status_code=None, error_code="validation_error")
        self.field = field


@dataclass
class SteamAuthConfig:
    """Configuration for Steam authentication client.

    Attributes:
        connect_timeout: Timeout for establishing connection (seconds).
        read_timeout: Timeout for reading response (seconds).
        max_retries: Maximum number of retry attempts for transient failures.
        retry_base_delay: Base delay between retries (seconds).
        retry_max_delay: Maximum delay between retries (seconds).
    """

    connect_timeout: float = 5.0
    read_timeout: float = 15.0
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 10.0


@dataclass
class SteamExchangeResult:
    """Result of a successful Steam ticket exchange."""

    token: str
    """The Player JWT to use for NPC chat requests."""

    expires_at: datetime
    """When the token expires (UTC)."""

    player_id: str
    """The Steam ID of the authenticated player."""

    studio_id: str
    """The studio ID the token is associated with."""

    title_id: str
    """The title ID the token is associated with."""

    @classmethod
    def from_response(cls, data: dict) -> "SteamExchangeResult":
        """Create from API response."""
        expires_str = data["expires_at"]
        # Ensure timezone-aware datetime
        if expires_str.endswith("Z"):
            expires_str = expires_str[:-1] + "+00:00"
        expires_at = datetime.fromisoformat(expires_str)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        return cls(
            token=data["token"],
            expires_at=expires_at,
            player_id=data["player_id"],
            studio_id=data["studio_id"],
            title_id=data["title_id"],
        )

    def is_expired(self) -> bool:
        """Check if the token has expired."""
        now_utc = datetime.now(timezone.utc)
        return now_utc >= self.expires_at

    def expires_in_seconds(self) -> float:
        """Get seconds until token expires (negative if expired)."""
        now_utc = datetime.now(timezone.utc)
        delta = self.expires_at - now_utc
        return delta.total_seconds()


def _validate_app_id(app_id: str) -> None:
    """Validate Steam AppID format.

    Args:
        app_id: The Steam AppID to validate.

    Raises:
        ValidationError: If the AppID is invalid.
    """
    if not app_id:
        raise ValidationError("Steam AppID is required", field="app_id")

    if not STEAM_APP_ID_PATTERN.match(app_id):
        raise ValidationError(
            f"Invalid Steam AppID format: '{app_id}'. Expected numeric value (e.g., '480')",
            field="app_id",
        )


def _validate_ticket(ticket: str) -> None:
    """Validate Steam session ticket format.

    Args:
        ticket: The base64-encoded Steam session ticket.

    Raises:
        ValidationError: If the ticket is invalid.
    """
    if not ticket:
        raise ValidationError("Steam session ticket is required", field="ticket")

    # Check for reasonable length (Steam tickets are typically 52-256 bytes when decoded)
    if len(ticket) < 20:
        raise ValidationError(
            "Steam session ticket is too short. Ensure it's properly encoded.",
            field="ticket",
        )

    if len(ticket) > 2048:
        raise ValidationError(
            "Steam session ticket is too long. Maximum 2048 characters.",
            field="ticket",
        )

    # Validate base64 encoding
    try:
        # Try decoding to verify it's valid base64
        decoded = base64.b64decode(ticket, validate=True)
        if len(decoded) < 10:
            raise ValidationError(
                "Decoded Steam ticket is too small. Verify the ticket is correct.",
                field="ticket",
            )
    except Exception as e:
        raise ValidationError(
            f"Invalid base64 encoding for Steam ticket: {e}",
            field="ticket",
        )


class SteamAuth:
    """Client for Steam authentication with Loreguard API.

    Exchanges Steam session tickets for Player JWTs that can be used
    for player-authenticated NPC chat.

    Features:
    - Input validation for AppID and tickets
    - Automatic retry with exponential backoff
    - Configurable timeouts
    - Structured logging

    Example:
        steam_auth = SteamAuth()

        # In your game client, get a Steam session ticket:
        # byte[] ticketData = SteamUser.GetAuthSessionTicket(...)
        # string ticketBase64 = Convert.ToBase64String(ticketData)

        result = await steam_auth.exchange_ticket(
            app_id="your_steam_app_id",
            ticket=ticketBase64
        )

        # Use result.token as the Authorization header for /api/chat
    """

    def __init__(
        self,
        api_url: str = LOREGUARD_API_URL,
        config: Optional[SteamAuthConfig] = None,
    ):
        """Initialize the Steam auth client.

        Args:
            api_url: The Loreguard API base URL.
            config: Optional configuration for timeouts and retries.
        """
        self.api_url = api_url.rstrip("/")
        self.config = config or SteamAuthConfig()

    def _get_timeout(self) -> httpx.Timeout:
        """Get configured timeout settings."""
        return httpx.Timeout(
            connect=self.config.connect_timeout,
            read=self.config.read_timeout,
            write=self.config.read_timeout,
            pool=self.config.connect_timeout,
        )

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt with exponential backoff and jitter.

        Args:
            attempt: The current attempt number (0-indexed).

        Returns:
            Delay in seconds before next retry.
        """
        # Exponential backoff: base_delay * 2^attempt
        delay = self.config.retry_base_delay * (2**attempt)
        # Cap at max delay
        delay = min(delay, self.config.retry_max_delay)
        # Add jitter (0-25% of delay)
        jitter = random.uniform(0, delay * 0.25)
        return delay + jitter

    def _is_retryable_error(self, status_code: int) -> bool:
        """Check if HTTP status code is retryable.

        Args:
            status_code: HTTP response status code.

        Returns:
            True if the request should be retried.
        """
        # Retry on server errors and rate limiting
        return status_code in (429, 500, 502, 503, 504)

    async def exchange_ticket(
        self, app_id: str, ticket: str
    ) -> SteamExchangeResult:
        """Exchange a Steam session ticket for a Player JWT.

        Args:
            app_id: Your Steam AppID (must be registered in Loreguard dashboard).
            ticket: Base64-encoded Steam session ticket from GetAuthSessionTicket().

        Returns:
            SteamExchangeResult with the Player JWT and metadata.

        Raises:
            ValidationError: If input validation fails.
            SteamAuthError: If authentication fails.
            RateLimitError: If rate limited (429).
        """
        # Validate inputs
        _validate_app_id(app_id)
        _validate_ticket(ticket)

        payload = {
            "app_id": str(app_id),
            "ticket": ticket,
        }

        last_error: Optional[Exception] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = await self._do_exchange(payload, attempt)
                return result

            except RateLimitError:
                # Don't retry rate limits - let caller handle backoff
                raise

            except SteamAuthError as e:
                # Don't retry client errors (4xx except 429)
                if e.status_code and 400 <= e.status_code < 500:
                    raise
                last_error = e

            except httpx.RequestError as e:
                # Network errors are retryable
                last_error = SteamAuthError(
                    "Network error during Steam authentication. Check your connection.",
                    status_code=None,
                    error_code="network_error",
                )
                logger.warning(
                    "Steam auth network error (attempt %d/%d): %s",
                    attempt + 1,
                    self.config.max_retries + 1,
                    str(e),
                )

            # Calculate retry delay if we have more attempts
            if attempt < self.config.max_retries:
                delay = self._calculate_retry_delay(attempt)
                logger.info(
                    "Retrying Steam auth in %.2f seconds (attempt %d/%d)",
                    delay,
                    attempt + 2,
                    self.config.max_retries + 1,
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        if last_error:
            raise last_error
        raise SteamAuthError(
            "Steam authentication failed after all retries",
            status_code=None,
            error_code="max_retries_exceeded",
        )

    async def _do_exchange(
        self, payload: dict, attempt: int
    ) -> SteamExchangeResult:
        """Perform the actual ticket exchange request.

        Args:
            payload: Request payload with app_id and ticket.
            attempt: Current attempt number for logging.

        Returns:
            SteamExchangeResult on success.

        Raises:
            SteamAuthError: On authentication failure.
            RateLimitError: On rate limiting.
            httpx.RequestError: On network failure.
        """
        async with httpx.AsyncClient(timeout=self._get_timeout()) as client:
            logger.debug(
                "Steam auth attempt %d for app_id=%s",
                attempt + 1,
                payload["app_id"],
            )

            response = await client.post(
                f"{self.api_url}/api/player/steam",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = SteamExchangeResult.from_response(response.json())
                logger.info(
                    "Steam auth successful: player_id=%s, expires_in=%ds",
                    result.player_id,
                    int(result.expires_in_seconds()),
                )
                return result

            # Parse error response
            try:
                error_data = response.json()
            except Exception:
                error_data = {}

            # Handle specific error cases
            if response.status_code == 429:
                logger.warning(
                    "Steam auth rate limited: %s/%s requests, resets at %s",
                    error_data.get("requests_used", "?"),
                    error_data.get("requests_limit", "?"),
                    error_data.get("reset_at", "?"),
                )
                raise RateLimitError(
                    message="Rate limit exceeded. Please wait before retrying.",
                    limit_type=error_data.get("limit_type", "unknown"),
                    requests_used=error_data.get("requests_used", 0),
                    requests_limit=error_data.get("requests_limit", 0),
                    reset_at=error_data.get("reset_at"),
                )

            if response.status_code == 400:
                logger.error(
                    "Steam auth validation error: %s",
                    error_data.get("error", "unknown"),
                )
                raise SteamAuthError(
                    "Invalid request. Check your AppID and ticket format.",
                    status_code=400,
                    error_code=error_data.get("code", "invalid_request"),
                )

            if response.status_code == 401:
                logger.error(
                    "Steam auth failed: %s",
                    error_data.get("error", "invalid ticket"),
                )
                raise SteamAuthError(
                    "Steam ticket validation failed. The ticket may be expired or invalid.",
                    status_code=401,
                    error_code=error_data.get("code", "invalid_ticket"),
                )

            if response.status_code == 404:
                logger.error(
                    "Steam app not found: app_id=%s",
                    payload["app_id"],
                )
                raise SteamAuthError(
                    "Steam AppID not registered. Add your app in the Loreguard dashboard.",
                    status_code=404,
                    error_code=error_data.get("code", "app_not_found"),
                )

            # Server errors (5xx) - will be retried
            if response.status_code >= 500:
                logger.warning(
                    "Steam auth server error: HTTP %d",
                    response.status_code,
                )
                raise SteamAuthError(
                    "Server error during authentication. Please try again.",
                    status_code=response.status_code,
                    error_code="server_error",
                )

            # Unexpected error
            logger.error(
                "Steam auth unexpected error: HTTP %d, body=%s",
                response.status_code,
                error_data,
            )
            raise SteamAuthError(
                "Unexpected error during Steam authentication.",
                status_code=response.status_code,
                error_code=error_data.get("code", "unknown_error"),
            )

    def exchange_ticket_sync(
        self, app_id: str, ticket: str
    ) -> SteamExchangeResult:
        """Synchronous version of exchange_ticket.

        Args:
            app_id: Your Steam AppID.
            ticket: Base64-encoded Steam session ticket.

        Returns:
            SteamExchangeResult with the Player JWT and metadata.
        """
        return asyncio.run(self.exchange_ticket(app_id, ticket))
