import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.http_server import EmbeddedHTTPServer
from src.config import LoreguardConfig
from src.runtime import RuntimeInfo


class RuntimeCredentialTest(unittest.TestCase):
    def test_runtime_file_is_user_only_and_preserves_token(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "loreguard"
            info = RuntimeInfo(
                port=12345,
                pid=os.getpid(),
                started_at="now",
                version="test",
                api_token="local-capability",
            )
            with patch("src.runtime.get_data_dir", return_value=data_dir):
                info.save()
                loaded = RuntimeInfo.load()

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.api_token, "local-capability")
            if os.name != "nt":
                self.assertEqual((data_dir / "runtime.json").stat().st_mode & 0o777, 0o600)
                self.assertEqual(data_dir.stat().st_mode & 0o777, 0o700)
            self.assertEqual(list(data_dir.glob(".runtime-*.json")), [])

    def test_sdk_server_uses_unique_tokens_and_loopback_only(self):
        first = EmbeddedHTTPServer(tunnel=None)
        second = EmbeddedHTTPServer(tunnel=None)
        self.assertGreaterEqual(len(first.api_token), 32)
        self.assertNotEqual(first.api_token, second.api_token)

        with self.assertRaisesRegex(ValueError, "loopback"):
            EmbeddedHTTPServer(tunnel=None, host="0.0.0.0")

    def test_saved_config_is_user_only_and_atomic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "loreguard"
            data_dir.mkdir()
            with patch("src.config.get_config_path", return_value=data_dir / "config.json"):
                LoreguardConfig(api_token="cloud-token").save()

            path = data_dir / "config.json"
            if os.name != "nt":
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(data_dir.glob(".config-*.json")), [])


if __name__ == "__main__":
    unittest.main()
