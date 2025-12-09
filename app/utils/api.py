import os
import re

from openai import OpenAI
from jinja2 import Environment, FileSystemLoader
from typing import List, Literal


def translate_text(base_url: str, authtoken: str, model: str, max_tokens: int, text: str, language: str = "en") -> str:
    """
    Translates a text into the specified language using a VLLM model.
    """
    client = OpenAI(api_key=authtoken, base_url=base_url)

    # Jinja2 environment for templates (chemin relatif, compatible déploiement)
    templates_dir = os.path.join(os.path.dirname(__file__), "..", "..", "templates")
    env = Environment(loader=FileSystemLoader(templates_dir))
    system_template = env.get_template("srt_translate_system_prompt.j2")
    user_template = env.get_template("srt_translate_user_prompt.j2")

    system_prompt = system_template.render(language=language)
    user_prompt = user_template.render(srt_text=text)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        max_tokens=max_tokens,
        temperature=0,
    )

    # Remove <think> tags from the response content
    content = re.sub(r"<think>.*?</think>", "", response.choices[0].message.content, flags=re.DOTALL)  # type: ignore

    return content


def format_summary(
    base_url: str,
    authtoken: str,
    model: str,
    max_tokens: int,
    temperature: float,
    summary: str,
    language: str = "en",
    prompt_template: str = "meeting_report_prompt.j2",
    custom_prompt: str | None = None,
) -> str:
    """
    Reformats a summary into a structured report using a VLLM instance.
    The prompt is in English, but the output is generated in the specified language.
    """
    client = OpenAI(api_key=authtoken, base_url=base_url)

    # Jinja2 environment for templates (chemin relatif, compatible déploiement)
    templates_dir = os.path.join(os.path.dirname(__file__), "..", "..", "templates")
    env = Environment(loader=FileSystemLoader(templates_dir))

    if custom_prompt:
        system_prompt = custom_prompt
    else:
        system_template = env.get_template(prompt_template)
        system_prompt = system_template.render(language=language)

    user_template = env.get_template("summary_user_prompt.j2")
    user_prompt = user_template.render(summary=summary)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Remove <think> tags from the response content
    content = re.sub(r"<think>.*?</think>", "", response.choices[0].message.content, flags=re.DOTALL)  # type: ignore

    return content


def transcribe_audio_via_api(
    base_url: str,
    authtoken: str,
    model: str,
    file_path: str,
    timestamp_granularities: List[Literal["word", "segment"]] = ["segment", "word"],
    language: str | None = None,
) -> dict:
    """
    Transcribes an audio file using an OpenAI-compatible API.
    """
    client = OpenAI(api_key=authtoken, base_url=base_url)

    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=timestamp_granularities,
            timeout=3600,  # 1 hour timeout
            language=language,
        )

    # The API returns a pydantic model, we need to convert it to a dict
    return transcription.model_dump()
