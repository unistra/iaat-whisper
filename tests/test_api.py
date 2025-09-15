from unittest.mock import Mock, patch
from utils.api import translate_text, format_summary # type: ignore

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
