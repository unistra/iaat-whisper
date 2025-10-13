from unittest import mock
from unittest.mock import Mock, patch
from utils.api import translate_text, format_summary, transcribe_audio_via_api # type: ignore
from jinja2 import Environment

def test_translate_text_success():
    input_text = """1
00:00:01,000 --> 00:00:03,000
Hello, how are you?

2
00:00:04,000 --> 00:00:06,000
I'm doing great!
"""

    expected_output = """1
00:00:01,000 --> 00:00:03,000
Bonjour, comment vas-tu ?

2
00:00:04,000 --> 00:00:06,000
Je vais très bien !
"""

    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content=expected_output))]
    mock_client.chat.completions.create.return_value = mock_response

    with patch('utils.api.OpenAI', return_value=mock_client):
        result = translate_text(
            base_url="http://example.com",
            authtoken="fake_token",
            model="fake_model",
            max_tokens=1024,
            text=input_text,
            language="fr"
        )

    assert result == expected_output
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args
    assert call_args[1]["model"] == "fake_model"
    assert "You are a professional subtitle translator." in call_args[1]["messages"][0]["content"]
    assert "French" in call_args[1]["messages"][0]["content"] or "fr" in call_args[1]["messages"][0]["content"].lower()


def test_format_summary_success():
    # Exemple de résumé brut
    input_summary = """The meeting started with introductions. We discussed project timelines. 
    A decision was made to extend the deadline. John will update the schedule."""

    # Réponse simulée attendue en français (Markdown structuré)
    expected_output = """# Rapport de réunion structuré

## Introduction
Le meeting a commencé par des introductions.

## Points
Nous avons discuté des délais du projet.

## Décisions
Une décision a été prise de prolonger la date limite.

## Actions
John mettra à jour le calendrier.
"""

    # Mock de l'objet OpenAI et de la réponse
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content=expected_output))]
    mock_client.chat.completions.create.return_value = mock_response

    # Patch de la classe OpenAI pour retourner notre mock
    with patch('utils.api.OpenAI', return_value=mock_client):
        result = format_summary(
            base_url="http://example.com",
            authtoken="fake_token",
            model="fake_model",
            max_tokens=1024,
            temperature=0.4,
            summary=input_summary,
            language="fr"
        )

    # Vérifications
    assert result == expected_output
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args
    assert call_args[1]["model"] == "fake_model"
    assert "You are a professional report writer." in call_args[1]["messages"][0]["content"]
    assert "FR" in call_args[1]["messages"][0]["content"] or "fr" in call_args[1]["messages"][1]["content"].lower()


def test_transcribe_audio_via_api_success():
    expected_output = {
        "text": "Hello, how are you?",
        "segments": [
            {
                "id": 1,
                "start": 0.0,
                "end": 3.0,
                "text": "Hello, how are you?",
            }
        ],
        "language": "en",
    }

    mock_client = Mock()
    mock_response = Mock()
    mock_response.model_dump.return_value = expected_output
    mock_client.audio.transcriptions.create.return_value = mock_response

    with patch('utils.api.OpenAI', return_value=mock_client):
        with patch('builtins.open', mock.mock_open(read_data=b'test_audio_data')) as mock_file:
            result = transcribe_audio_via_api(
                base_url="http://example.com",
                authtoken="fake_token",
                model="fake_model",
                file_path="fake_path.wav",
            )

    assert result == expected_output
    mock_client.audio.transcriptions.create.assert_called_once()
    call_args = mock_client.audio.transcriptions.create.call_args
    assert call_args[1]["model"] == "fake_model"
    assert call_args[1]["response_format"] == "verbose_json"
    assert call_args[1]["timestamp_granularities"] == ["segment", "word"]


def test_format_summary_with_prompt_template():
    input_summary = "This is a presentation about AI."
    expected_output = "## MAIN TOPIC\nAI\n\n## KEY POINTS\n- Point 1\n- Point 2"
    prompt_template_name = "presentation_summary_prompt.j2"

    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content=expected_output))]
    mock_client.chat.completions.create.return_value = mock_response

    # Mock Jinja2 environment
    mock_env = Mock()
    mock_template = Mock()
    mock_env.get_template.return_value = mock_template
    mock_template.render.return_value = "rendered_prompt"

    with patch('utils.api.OpenAI', return_value=mock_client):
        with patch('utils.api.Environment', return_value=mock_env):
            result = format_summary(
                base_url="http://example.com",
                authtoken="fake_token",
                model="fake_model",
                max_tokens=1024,
                temperature=0.4,
                summary=input_summary,
                language="en",
                prompt_template=prompt_template_name
            )

    # Check that get_template was called with the correct template name for system prompt
    # and also for the user prompt
    mock_env.get_template.assert_any_call(prompt_template_name)
    mock_env.get_template.assert_any_call("summary_user_prompt.j2")

    # Check that the result is correct
    assert result == expected_output