import os


def get_secrets():
    """
    Return a dictionary containing secrets retrieved from environment variables.
    """
    return {
        "auth": {
            "redirect_uri": os.getenv("REDIRECT_URI"),
            "cookie_secret": os.getenv("COOKIE_SECRET"),
            "client_id": os.getenv("CLIENT_ID"),
            "client_secret": os.getenv("CLIENT_SECRET"),
            "server_metadata_url": os.getenv("SERVER_METADATA_URL"),
            "client_kwargs": {"prompt": os.getenv("CLIENT_PROMPT")},
        },
        "huggingface": {"token": os.getenv("HUGGINGFACE_ACCESS_TOKEN")},
        "llm": {
            "url": os.getenv("LLM_URL"),
            "token": os.getenv("LLM_TOKEN"),
            "model": os.getenv("LLM_MODEL"),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "32768")),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.4")),
        },
        "app": {
            "use_custom_style": os.getenv("USE_CUSTOM_STYLE", "false").lower() == "true",
            "sumy_length_default": int(os.getenv("SUMY_LENGTH_DEFAULT", "80")),
        }
    }
