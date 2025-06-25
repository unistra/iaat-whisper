from openai import OpenAI
from jinja2 import Environment, FileSystemLoader
import os
import re


def translate_text(base_url: str, authtoken: str, model: str, text: str, language: str = "en") -> str:
    """
    Translates a text into the specified language using a VLLM model.
    """
    client = OpenAI(api_key=authtoken, base_url=base_url)

    # Jinja2 environment for templates (chemin relatif, compatible déploiement)
    templates_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'templates')
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
        max_tokens=16384,
        temperature=0.2,
    )

    # Remove <think> tags from the response content
    content = re.sub(r'<think>.*?</think>', '', response.choices[0].message.content, flags=re.DOTALL)   # type: ignore

    return content


def format_summary(base_url: str, authtoken: str, model: str, summary: str, language="en") -> str:
    """
    Reformats a summary into a structured meeting report using a VLLM instance.
    The prompt is in English, but the output is generated in the specified language.
    """
    client = OpenAI(api_key=authtoken, base_url=base_url)

    # Jinja2 environment for templates (chemin relatif, compatible déploiement)
    templates_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'templates')
    env = Environment(loader=FileSystemLoader(templates_dir))
    system_template = env.get_template("summary_system_prompt.j2")
    user_template = env.get_template("summary_user_prompt.j2")

    system_prompt = system_template.render(language=language)
    user_prompt = user_template.render(summary=summary)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        max_tokens=16384,
        temperature=0.4,
    )

    # Remove <think> tags from the response content
    content = re.sub(r'<think>.*?</think>', '', response.choices[0].message.content, flags=re.DOTALL)   # type: ignore

    return content
