import unittest
from unittest.mock import MagicMock, patch

from chatml.gemini_connector import GeminiConnector


class TestGeminiConnector(unittest.TestCase):
    def setUp(self):
        self.connector = GeminiConnector()

    @patch("chatml.gemini_connector.genai.list_models")
    @patch("chatml.gemini_connector.genai.GenerativeModel")
    @patch(
        "chatml.gemini_connector.DEFAULT_CHAT_MODEL",
        "gemini-3-flash-preview",
        create=True,
    )
    def test_get_model_uses_default_chat_model(self, mock_gen_model, mock_list_models):
        """Tests if get_model selects the model defined in DEFAULT_CHAT_MODEL."""
        # Setup mock models
        mock_other = MagicMock()
        mock_other.name = "models/gemini-1.5-flash"
        mock_other.supported_generation_methods = ["generateContent"]

        mock_target = MagicMock()
        mock_target.name = "models/gemini-3-flash-preview"
        mock_target.supported_generation_methods = ["generateContent"]

        # Current logic picks the first "flash" model, which would be gemini-1.5-flash
        mock_list_models.return_value = [mock_other, mock_target]

        # Call get_model
        self.connector.get_model()

        # Verify GenerativeModel was initialized with the correct name
        mock_gen_model.assert_called_once()
        args, _ = mock_gen_model.call_args
        self.assertEqual(args[0], "models/gemini-3-flash-preview")

    @patch("chatml.gemini_connector.genai.list_models")
    @patch("chatml.gemini_connector.genai.GenerativeModel")
    @patch(
        "chatml.gemini_connector.DEFAULT_CHAT_MODEL",
        "gemini-3-flash-preview",
        create=True,
    )
    def test_get_model_fallbacks_and_warns_when_default_unavailable(
        self, mock_gen_model, mock_list_models
    ):
        """Tests if get_model falls back to another flash model and warns when the default is missing."""
        # Setup mock models - default is missing, but another flash is present
        mock_flash = MagicMock()
        mock_flash.name = "models/gemini-1.5-flash"
        mock_flash.supported_generation_methods = ["generateContent"]
        mock_list_models.return_value = [mock_flash]

        with patch.object(self.connector.console, "print") as mock_print:
            self.connector.get_model()

        # Verify fallback
        mock_gen_model.assert_called_once()
        args, _ = mock_gen_model.call_args
        self.assertEqual(args[0], "models/gemini-1.5-flash")

        # Verify warning was printed
        warning_called = any(
            "warning" in str(call).lower() and "gemini-3-flash-preview" in str(call)
            for call in mock_print.call_args_list
        )
        self.assertTrue(warning_called, "A warning message should have been printed.")


if __name__ == "__main__":
    unittest.main()
