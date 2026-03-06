"""Tests for olmlx.__main__."""

from unittest.mock import patch

from olmlx.__main__ import main


class TestMain:
    def test_main_delegates_to_cli_main(self):
        with patch("olmlx.__main__.cli_main") as mock_cli:
            main()
            mock_cli.assert_called_once()
