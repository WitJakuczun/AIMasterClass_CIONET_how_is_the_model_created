import unittest
from unittest.mock import patch


class TestChatEntry(unittest.TestCase):
    @patch("chatml.chat.main")
    @patch("asyncio.run")
    def test_run_chat_calls_asyncio_run_with_main(self, mock_run, mock_main):
        """Tests if run_chat correctly initializes the asyncio loop with the main function."""
        # This should fail initially because run_chat doesn't exist
        from chatml.chat import run_chat

        run_chat()

        # Verify asyncio.run was called
        mock_run.assert_called_once()

        # Verify that main was called
        mock_main.assert_called_once()

    def test_pyproject_entry_point(self):
        """Verifies that pyproject.toml has the correct entry point for chatml."""
        import os
        import tomllib

        pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        scripts = data.get("project", {}).get("scripts", {})
        self.assertEqual(scripts.get("chatml"), "chatml.chat:run_chat")


if __name__ == "__main__":
    unittest.main()
