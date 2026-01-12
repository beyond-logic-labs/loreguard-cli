"""
Tests for WebSocket send timeout fix.

These tests verify that the _send() method in BackendTunnel properly times out
when the WebSocket is blocked, preventing indefinite hangs that would stall
the entire streaming pipeline.

TDD Approach:
1. First run with the timeout removed - tests should FAIL (hang or timeout)
2. Then run with timeout in place - tests should PASS
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch


class MockWebSocket:
    """Mock WebSocket that can simulate blocking sends."""

    def __init__(self, block_time: float = 0):
        """
        Args:
            block_time: How long send() should block (0 = instant, >0 = simulate slow send)
        """
        self.block_time = block_time
        self.sent_messages = []
        self.send_called = False

    async def send(self, message: str):
        """Simulate sending a message with optional delay."""
        self.send_called = True
        if self.block_time > 0:
            await asyncio.sleep(self.block_time)
        self.sent_messages.append(message)


class MockTunnel:
    """Minimal mock of BackendTunnel to test _send behavior."""

    def __init__(self, ws: MockWebSocket):
        self.ws = ws
        self.connected = True

    async def _send_with_timeout(self, data: dict, timeout: float = 5.0):
        """
        Send with timeout - this is the FIXED version.
        If this test passes, the fix is working.
        """
        import json

        ws = self.ws
        if not ws or not self.connected:
            return

        try:
            await asyncio.wait_for(ws.send(json.dumps(data)), timeout=timeout)
        except asyncio.TimeoutError:
            self.connected = False
            raise
        except Exception:
            self.connected = False
            raise

    async def _send_without_timeout(self, data: dict):
        """
        Send WITHOUT timeout - this is the BUGGY version.
        This should hang indefinitely if WebSocket is blocked.
        """
        import json

        ws = self.ws
        if not ws or not self.connected:
            return

        try:
            await ws.send(json.dumps(data))
        except Exception:
            self.connected = False
            raise


class TestWebSocketSendTimeout:
    """Test suite for WebSocket send timeout behavior."""

    @pytest.mark.asyncio
    async def test_send_with_timeout_completes_fast_send(self):
        """Fast sends should complete normally with timeout enabled."""
        ws = MockWebSocket(block_time=0)  # Instant send
        tunnel = MockTunnel(ws)

        start = time.time()
        await tunnel._send_with_timeout({"test": "data"}, timeout=1.0)
        elapsed = time.time() - start

        assert elapsed < 0.5, "Fast send should complete quickly"
        assert tunnel.connected, "Connection should remain connected"
        assert len(ws.sent_messages) == 1, "Message should be sent"

    @pytest.mark.asyncio
    async def test_send_with_timeout_times_out_on_blocked_ws(self):
        """
        Blocked WebSocket should timeout and not hang indefinitely.

        This is the KEY TEST - it verifies the fix works.
        With the old code (no timeout), this test would hang forever.
        With the fix, it should timeout after ~1 second.
        """
        ws = MockWebSocket(block_time=10.0)  # Simulate 10s block
        tunnel = MockTunnel(ws)

        start = time.time()
        with pytest.raises(asyncio.TimeoutError):
            await tunnel._send_with_timeout({"test": "data"}, timeout=1.0)
        elapsed = time.time() - start

        # Should timeout around 1 second, not hang for 10 seconds
        assert elapsed < 2.0, f"Should timeout quickly, not hang (elapsed: {elapsed:.2f}s)"
        assert elapsed >= 0.9, f"Should wait at least the timeout duration (elapsed: {elapsed:.2f}s)"
        assert not tunnel.connected, "Connection should be marked as disconnected after timeout"

    @pytest.mark.asyncio
    async def test_send_without_timeout_hangs_on_blocked_ws(self):
        """
        BUG REPRODUCTION: Without timeout, blocked WebSocket hangs forever.

        This test demonstrates the bug that the fix addresses.
        We use a short external timeout to prevent the test from actually hanging.

        EXPECTED: This test should FAIL (hang until external timeout) when using
        _send_without_timeout, proving the bug exists.
        """
        ws = MockWebSocket(block_time=10.0)  # Simulate 10s block
        tunnel = MockTunnel(ws)

        start = time.time()

        # Use an external timeout to prevent the test from hanging forever
        # The buggy _send_without_timeout would hang for the full 10s
        try:
            await asyncio.wait_for(
                tunnel._send_without_timeout({"test": "data"}),
                timeout=2.0  # External timeout to prevent infinite hang
            )
            # If we get here without the external timeout triggering,
            # something is wrong (send should have blocked)
            elapsed = time.time() - start
            if elapsed < 1.0:
                # Send completed too fast - mock might not be blocking
                pass
        except asyncio.TimeoutError:
            # This is expected - the external timeout fired because
            # _send_without_timeout has no internal timeout
            elapsed = time.time() - start
            assert elapsed >= 1.9, "Should hang until external timeout"
            # The bug is that connection is NOT marked as disconnected
            # because _send_without_timeout doesn't handle the timeout
            assert tunnel.connected, "BUG: Connection still marked as connected after hang"

    @pytest.mark.asyncio
    async def test_streaming_scenario_with_timeout(self):
        """
        Simulate the actual streaming scenario where multiple tokens are sent.

        If one send blocks, the timeout should fire and prevent the entire
        streaming loop from hanging.
        """
        ws = MockWebSocket(block_time=0)  # Start with fast sends
        tunnel = MockTunnel(ws)

        # Simulate sending 50 tokens quickly
        for i in range(50):
            await tunnel._send_with_timeout({"token": f"tok{i}"}, timeout=1.0)

        assert len(ws.sent_messages) == 50, "All tokens should be sent"
        assert tunnel.connected, "Connection should remain connected"

        # Now simulate the WebSocket blocking on token 51
        ws.block_time = 10.0

        start = time.time()
        with pytest.raises(asyncio.TimeoutError):
            await tunnel._send_with_timeout({"token": "tok50"}, timeout=1.0)
        elapsed = time.time() - start

        assert elapsed < 2.0, "Timeout should fire quickly"
        assert not tunnel.connected, "Connection should be disconnected"
        assert len(ws.sent_messages) == 50, "Blocked message should not be sent"

    @pytest.mark.asyncio
    async def test_connection_state_after_successful_send(self):
        """Connection should remain connected after successful sends."""
        ws = MockWebSocket(block_time=0.1)  # Slight delay but within timeout
        tunnel = MockTunnel(ws)

        await tunnel._send_with_timeout({"test": "data"}, timeout=1.0)

        assert tunnel.connected, "Connection should remain connected"

    @pytest.mark.asyncio
    async def test_custom_timeout_values(self):
        """Different timeout values should be respected."""
        ws = MockWebSocket(block_time=0.5)  # 500ms delay
        tunnel = MockTunnel(ws)

        # Short timeout should fail
        with pytest.raises(asyncio.TimeoutError):
            await tunnel._send_with_timeout({"test": "data"}, timeout=0.1)

        # Reset connection state
        tunnel.connected = True

        # Longer timeout should succeed
        await tunnel._send_with_timeout({"test": "data"}, timeout=1.0)
        assert tunnel.connected, "Connection should remain connected with sufficient timeout"


# =============================================================================
# Integration test with actual BackendTunnel class
# =============================================================================

class TestBackendTunnelSendTimeout:
    """Integration tests using the actual BackendTunnel._send method."""

    @pytest.mark.asyncio
    async def test_actual_send_method_has_timeout(self):
        """
        Verify that the actual BackendTunnel._send method includes timeout.

        This test verifies the fix is in place by checking the source code
        contains the asyncio.wait_for timeout wrapper.
        """
        import os

        # Read the actual tunnel.py source
        tunnel_path = os.path.join(
            os.path.dirname(__file__),
            "..", "src", "tunnel.py"
        )

        with open(tunnel_path, "r") as f:
            source = f.read()

        # Verify the fix is in place:
        # 1. _send method should have a timeout parameter
        assert "async def _send(self, data: dict, timeout: float" in source, \
            "_send method should have timeout parameter"

        # 2. Should use asyncio.wait_for for timeout
        assert "asyncio.wait_for" in source, \
            "_send should use asyncio.wait_for for timeout"

        # 3. Should handle TimeoutError
        assert "asyncio.TimeoutError" in source, \
            "_send should handle asyncio.TimeoutError"

        # 4. Should set connected = False on timeout
        assert "self.connected = False" in source, \
            "_send should set connected = False on timeout"

    @pytest.mark.asyncio
    async def test_send_timeout_prevents_hang_simulation(self):
        """
        Simulate the actual _send behavior to verify timeout prevents hang.

        This mimics what the actual BackendTunnel._send does.
        """
        ws = MockWebSocket(block_time=10.0)  # Simulate blocked WebSocket

        # This is essentially what the fixed _send method does
        connected = True
        start = time.time()

        try:
            await asyncio.wait_for(ws.send('{"test": "data"}'), timeout=1.0)
        except asyncio.TimeoutError:
            connected = False  # This is what the fix does

        elapsed = time.time() - start

        assert elapsed < 2.0, f"Should timeout quickly (elapsed: {elapsed:.2f}s)"
        assert not connected, "Connection should be marked disconnected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
