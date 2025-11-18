import copy
import subprocess
import re

from functools import cache
from typing import Any
from utils.api import translate_text


def download_nltk_resources() -> None:
    """
    Download the necessary NLTK resources for tokenization.
    """
    import nltk

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


@cache
def get_random_name(speaker: str) -> str:
    """
    Return a random name for the speaker.
    """
    names = [
        "Alice",
        "Bob",
        "Charlie",
        "David",
        "Emma",
        "Fanny",
        "Gaston",
        "Hugo",
        "Isabelle",
        "Jules",
    ]
    index = int(speaker.split("_")[-1])
    return names[index] if index < len(names) else f"Locuteur {index}"


def convert_and_resample_audio(input_path: str, output_path: str, target_sr=16000) -> None:
    """Load an audio file (MP3, WAV, etc.), convert it to WAV 16 kHz, and save it."""
    import torchaudio
    import torchaudio.transforms as T

    waveform, sr = torchaudio.load(input_path)
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    torchaudio.save(output_path, waveform, target_sr)


def summarize_text(text: str, num_sentences=5, language="french") -> str:
    """
    Summarize a text using LexRank.
    """
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer

    try:
        tokenizer = Tokenizer(language)
    except Exception:
        raise ValueError(f"Langue non supportée par le tokenizer : {language}")

    parser = PlaintextParser.from_string(text, tokenizer)
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)


def assign_speakers(transcription: dict, diarization: Any) -> dict:
    """
    Returns a copy of the transcription with speakers assigned based on diarization.
    Assigns the speaker with the highest overlap to each transcription segment.
    """
    new_transcription = copy.deepcopy(transcription)  # Copie complète

    for seg in new_transcription["segments"]:
        max_overlap = 0
        assigned_speaker = None
        seg_start, seg_end = seg["start"], seg["end"]
        seg_duration = seg_end - seg_start

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Ignore les tours sans chevauchement possible
            if turn.end < seg_start:
                continue
            if turn.start > seg_end:
                break  # les suivants ne peuvent plus chevaucher (tours triés dans le temps)

            # Calcul du chevauchement
            overlap_start = max(seg_start, turn.start)
            overlap_end = min(seg_end, turn.end)
            overlap_duration = max(0, overlap_end - overlap_start)

            # Vérifie si le segment chevauche au moins 50% de sa durée avec ce locuteur
            overlap_ratio = overlap_duration / seg_duration

            # On garde le locuteur avec le plus grand chevauchement
            if overlap_ratio > max_overlap:
                max_overlap = overlap_ratio
                assigned_speaker = speaker

        # Si un locuteur a un chevauchement significatif, on l'assigne
        if assigned_speaker:
            seg["speaker"] = get_random_name(assigned_speaker)
        else:
            seg["speaker"] = "Inconnu"

    return new_transcription


def extract_audio_from_video(video_path: str, audio_path: str):
    """
    Extracts the audio from a video and saves it as a WAV file.
    """
    try:
        command = [
            "ffmpeg",
            "-i",
            video_path,  # Fichier vidéo en entrée
            "-ac",
            "1",  # Convertir en mono
            "-ar",
            "16000",  # Rééchantillonner en 16 kHz
            "-vn",  # Supprimer la vidéo
            "-y",  # Écraser le fichier si nécessaire
            audio_path,  # Fichier audio en sortie
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Erreur FFmpeg : {e.stderr.decode()}") from e


def split_srt(srt_text: str, max_chars: int) -> list[str]:
    """
    Divide an SRT file into text chunks that respect the SRT structure.
    """
    blocks = re.split(r"\n\n", srt_text.strip())
    chunks = []
    current_chunk = []

    for block in blocks:
        if not block.strip():  # Ignore les blocs vides
            continue

        # Si le bloc entier dépasse la limite, on commence un nouveau chunk pour ce bloc
        if sum(len(b) for b in current_chunk) + len(block) + 2 > max_chars:
            if current_chunk:  # Ajoute uniquement si le chunk n'est pas vide
                chunks.append("\n\n".join(current_chunk))
            current_chunk = [block]
        else:
            current_chunk.append(block)

    if current_chunk:  # Ajoute le dernier chunk
        chunks.append("\n\n".join(current_chunk))

    return chunks


def translate_srt_in_chunks(
    base_url: str,
    authtoken: str,
    model: str,
    max_tokens: int,
    srt_text: str,
    language: str = "en",
    max_chars: int = 500,
) -> str:
    """
    Translate an SRT file in chunks to avoid a too long prompt.
    """
    chunks = split_srt(srt_text, max_chars)
    translated_chunks = []

    for chunk in chunks:
        translated_chunk = translate_text(base_url, authtoken, model, max_tokens, chunk, language)
        translated_chunks.append(translated_chunk)

    return "\n\n".join(translated_chunks)
