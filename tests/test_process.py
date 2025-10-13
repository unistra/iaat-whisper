from utils.process import (  # type: ignore
    get_random_name,
    download_nltk_resources,
    summarize_text,
    convert_and_resample_audio,
    assign_speakers,
    extract_audio_from_video,
    split_srt,
    translate_srt_in_chunks,
)
import pytest
import nltk
import torchaudio
import os
import torch
import subprocess
from unittest.mock import patch


@pytest.mark.parametrize(
    "speaker, expected",
    [
        ("spk_0", "Alice"),
        ("spk_3", "David"),
        ("spk_9", "Jules"),
        ("spk_10", "Locuteur 10"),
        ("spk_99", "Locuteur 99"),
    ],
)
def test_get_random_name(speaker, expected):
    assert get_random_name(speaker) == expected


def test_download_nltk_resources(mocker):
    mocker.patch("nltk.download")
    download_nltk_resources()
    nltk.download.assert_any_call("punkt", quiet=True)  # type: ignore
    nltk.download.assert_any_call("punkt_tab", quiet=True)  # type: ignore


def test_summarize_text():
    result = summarize_text("Bonjour, comment ça va ?", num_sentences=1, language="french")
    assert result == "Bonjour, comment ça va ?"


@pytest.fixture
def sample_audio(tmp_path):
    sample_rate = 4410
    duration = 1.0
    freq = 1000
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = 0.5 * torch.sin(2 * torch.pi * freq * t).unsqueeze(0)  # Mono
    input_path = tmp_path / "test_audio.wav"
    torchaudio.save(str(input_path), waveform, sample_rate)
    yield str(input_path)
    os.remove(str(input_path))


def test_convert_and_resample_audio(sample_audio, tmp_path):
    output_path = str(tmp_path / "resampled_audio.wav")
    convert_and_resample_audio(sample_audio, output_path, target_sr=16000)
    assert os.path.exists(output_path)
    waveform, sr = torchaudio.load(output_path)
    assert sr == 16000  # 16 kHz
    assert waveform.shape[0] == 1  # Mono
    assert waveform.numel() > 0  # Non empty file


class FakeTurn:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class FakeDiarization:
    def __init__(self, turns):
        self.turns = turns

    def itertracks(self, yield_label=False):
        for turn, speaker in self.turns:
            yield turn, None, speaker if yield_label else turn


def test_assign_speakers():
    transcription = {
        "segments": [
            {"start": 0.0, "end": 3.0, "text": "Bonjour"},
            {"start": 3.5, "end": 6.0, "text": "Comment ça va ?"},
            {"start": 6.5, "end": 9.0, "text": "Très bien, merci !"},
        ]
    }

    diarization = FakeDiarization(
        [
            (FakeTurn(0.0, 2.5), "spk_0"),
            (FakeTurn(3.0, 6.5), "spk_1"),
            (FakeTurn(6.0, 9.5), "spk_0"),
        ]
    )

    result = assign_speakers(transcription, diarization)
    assert result != transcription
    assert result["segments"][0]["speaker"] == get_random_name("spk_0")
    assert result["segments"][1]["speaker"] == get_random_name("spk_1")
    assert result["segments"][2]["speaker"] == get_random_name("spk_0")


def test_extract_audio_success():
    with patch("subprocess.run") as mock_run:
        extract_audio_from_video("video.mp4", "audio.wav")
        mock_run.assert_called_once_with(
            ["ffmpeg", "-i", "video.mp4", "-ac", "1", "-ar", "16000", "-vn", "-y", "audio.wav"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


def test_extract_audio_ffmpeg_error():
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg", stderr=b"Fake error")):
        with pytest.raises(RuntimeError, match="Erreur FFmpeg : Fake error"):
            extract_audio_from_video("video.mp4", "audio.wav")


def test_split_srt():
    srt_text = """1
00:00:01,000 --> 00:00:04,000
Bonjour !

2
00:00:05,000 --> 00:00:08,000
Comment ça va ?

3
00:00:09,000 --> 00:00:12,000
Très bien, merci !"""

    # Test with a large limit (no split)
    chunks = split_srt(srt_text, max_chars=1000)
    assert len(chunks) == 1
    assert chunks[0] == srt_text

    # Test with a limit that forces splitting
    chunks = split_srt(srt_text, max_chars=50)
    assert len(chunks) > 1

    # Verify that chunks do not exceed the limit
    for chunk in chunks:
        assert len(chunk) <= 50

    # Test with a single line SRT
    single_srt = "1\n00:00:01,000 --> 00:00:04,000\nTexte court."
    chunks = split_srt(single_srt, max_chars=10)
    assert len(chunks) == 1  # Un seul bloc car court
    assert chunks[0] == single_srt

    # Test with an empty SRT
    empty_srt = ""
    chunks = split_srt(empty_srt, max_chars=100)
    assert chunks == []


def test_translate_srt_in_chunks():
    srt_text = """1
00:00:01,000 --> 00:00:04,000
Bonjour !

2
00:00:05,000 --> 00:00:08,000
Comment ça va ?

3
00:00:09,000 --> 00:00:12,000
Très bien, merci !"""

    expected_chunks = [
        "1\n00:00:01,000 --> 00:00:04,000\nBonjour !",
        "2\n00:00:05,000 --> 00:00:08,000\nComment ça va ?",
        "3\n00:00:09,000 --> 00:00:12,000\nTrès bien, merci !",
    ]

    with patch("utils.process.translate_text") as mock_translate_text:
        mock_translate_text.side_effect = lambda base_url, authtoken, model, max_tokens, text, language: f"Translated: {text}"

        translate_srt_in_chunks("http://example.com", "dummy_token", "mistral", 1024, srt_text, "en", 50)

        for chunk in expected_chunks:
            mock_translate_text.assert_any_call("http://example.com", "dummy_token", "mistral", 1024, chunk, "en")
