"""ASGI request-body limits that apply before framework body parsing."""

from collections.abc import Awaitable, Callable
from typing import Any

from starlette.responses import JSONResponse


ASGIApp = Callable[
    [
        dict[str, Any],
        Callable[[], Awaitable[dict[str, Any]]],
        Callable[[dict[str, Any]], Awaitable[None]],
    ],
    Awaitable[None],
]


class RequestBodyLimitMiddleware:
    """Reject oversized bodies while consuming ASGI receive frames.

    FastAPI/Pydantic reads and buffers endpoint bodies before the handler runs.
    Reading the receive frames here bounds that buffering even when a client
    omits Content-Length and streams the request with chunked transfer encoding.
    """

    def __init__(self, app: ASGIApp, max_body_size: int) -> None:
        self.app = app
        self.max_body_size = max_body_size

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        messages: list[dict[str, Any]] = []
        total = 0
        while True:
            message = await receive()
            messages.append(message)
            if message.get("type") == "http.disconnect":
                break
            if message.get("type") != "http.request":
                continue

            total += len(message.get("body", b""))
            if total > self.max_body_size:
                response = JSONResponse(
                    status_code=413,
                    content={"error": "Request body too large"},
                )
                await response(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        message_index = 0

        async def replay_receive() -> dict[str, Any]:
            nonlocal message_index
            if message_index < len(messages):
                message = messages[message_index]
                message_index += 1
                return message
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)
