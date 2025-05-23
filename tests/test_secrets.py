import os
from unittest import mock
from utils.secrets import get_secrets # type: ignore

def test_get_secrets():
    with mock.patch.dict(os.environ, {
        "REDIRECT_URI": "https://example.com/redirect",
        "COOKIE_SECRET": "cookie_secret_value",
        "CLIENT_ID": "client_id_value",
        "CLIENT_SECRET": "client_secret_value",
        "SERVER_METADATA_URL": "https://example.com/metadata",
        "CLIENT_PROMPT": "client_prompt_value",
        "HUGGINGFACE_ACCESS_TOKEN": "huggingface_token",
        "LLM_URL": "https://llm.example.com",
        "LLM_TOKEN": "llm_token_value",
        "LLM_MODEL": "llm_model_value",
        "USE_CUSTOM_STYLE": "true",
    }):
        secrets = get_secrets()

        assert secrets["auth"]["redirect_uri"] == "https://example.com/redirect"
        assert secrets["auth"]["cookie_secret"] == "cookie_secret_value"
        assert secrets["auth"]["client_id"] == "client_id_value"
        assert secrets["auth"]["client_secret"] == "client_secret_value"
        assert secrets["auth"]["server_metadata_url"] == "https://example.com/metadata"
        assert secrets["auth"]["client_kwargs"]["prompt"] == "client_prompt_value"
        assert secrets["huggingface"]["token"] == "huggingface_token"
        assert secrets["llm"]["url"] == "https://llm.example.com"
        assert secrets["llm"]["token"] == "llm_token_value"
        assert secrets["llm"]["model"] == "llm_model_value"
        assert secrets["app"]["use_custom_style"] is True
