"""CLI dev-mode / token-precedence tests.

Covers security test plan CLIENT row:
  TC-53  Dev mode does not override an explicitly supplied worker token, does not
         invent a localhost/placeholder token, and still requires authentication.

Exercises the real argument-parsing and precedence logic in src/cli.py's main().
LoreguardCLI construction and asyncio.run are stubbed so main() runs to the point
where it hands the resolved token to the client, without touching the network,
llama-server, or models.
"""

import os
import sys
import unittest
from unittest.mock import patch

import src.cli as cli


class _RecordingCLI:
    """Captures the kwargs main() passes to LoreguardCLI."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _RecordingCLI.instances.append(self)

    def run(self):  # asyncio.run is stubbed, so this is never awaited
        return None


def _clean_env(**overrides):
    """os.environ with every LOREGUARD_* var removed, then overrides applied."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOREGUARD")}
    env.update(overrides)
    return env


class CliDevModeTokenTest(unittest.TestCase):
    def setUp(self):
        _RecordingCLI.instances = []

    def _run_main(self, argv, env):
        with patch.dict(os.environ, env, clear=True), patch.object(
            sys, "argv", ["loreguard"] + argv
        ), patch.object(cli, "LoreguardCLI", _RecordingCLI), patch.object(
            cli.asyncio, "run", return_value=0
        ):
            with self.assertRaises(SystemExit) as ctx:
                cli.main()
        return ctx.exception

    # --- TC-53: explicit token preserved under --dev -------------------------

    def test_dev_mode_preserves_explicit_token(self):
        exc = self._run_main(
            ["--dev", "--token", "lg_explicit_real_token", "--model-id", "qwen3-4b"],
            _clean_env(),
        )
        self.assertEqual(exc.code, 0)
        self.assertEqual(len(_RecordingCLI.instances), 1)
        token = _RecordingCLI.instances[0].kwargs["token"]
        self.assertEqual(token, "lg_explicit_real_token")
        # Dev mode must not swap in a placeholder / mock token.
        self.assertNotEqual(token, "dev_mock_token")

    def test_dev_mode_invents_no_localhost_placeholder(self):
        self._run_main(
            ["--dev", "--token", "lg_explicit_real_token", "--model-id", "qwen3-4b"],
            _clean_env(),
        )
        kwargs = _RecordingCLI.instances[0].kwargs
        token = kwargs["token"]
        self.assertNotIn("localhost", token)
        self.assertNotIn("127.0.0.1", token)
        # worker_id is left to hostname resolution (None), not a fabricated value.
        self.assertIsNone(kwargs["worker_id"])

    # --- TC-53: auth still required (no token invented) ----------------------

    def test_dev_mode_without_token_exits_and_builds_no_client(self):
        exc = self._run_main(["--dev", "--model-id", "qwen3-4b"], _clean_env())
        self.assertEqual(exc.code, 1)
        self.assertEqual(_RecordingCLI.instances, [])

    # --- token precedence -----------------------------------------------------

    def test_explicit_flag_overrides_env_token(self):
        self._run_main(
            ["--dev", "--token", "lg_cli_token", "--model-id", "qwen3-4b"],
            _clean_env(LOREGUARD_TOKEN="lg_env_token"),
        )
        self.assertEqual(_RecordingCLI.instances[0].kwargs["token"], "lg_cli_token")

    def test_env_token_used_when_no_flag(self):
        self._run_main(
            ["--dev", "--model-id", "qwen3-4b"],
            _clean_env(LOREGUARD_TOKEN="lg_env_token"),
        )
        self.assertEqual(_RecordingCLI.instances[0].kwargs["token"], "lg_env_token")


if __name__ == "__main__":
    unittest.main()
