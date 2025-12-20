from config import DEFAULT_CHAT_MODEL


def test_default_chat_model_is_set():
    """Verifies that the default chat model is correctly configured."""
    assert DEFAULT_CHAT_MODEL == "gemini-3-flash-preview"
