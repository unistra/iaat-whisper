import streamlit as st
from utils.log import logger, setup_logger
import whisper
import tempfile
import json
import os
import mimetypes
from streamlit.runtime.secrets import secrets_singleton
import torch
from typing import Any
from pyannote.audio import Pipeline
from utils.style import custom_font
from utils.process import (
    WHISPER_MODEL_OPTIONS,
    download_nltk_resources,
    convert_and_resample_audio,
    summarize_text,
    assign_speakers,
)
from utils.api import format_summary
from utils.secrets import get_secrets

# Setup logger
setup_logger()

download_nltk_resources()

# Secrets management
secrets_path = ".streamlit/secrets.toml"
if not os.path.exists(secrets_path):
    secrets_singleton._secrets = get_secrets()

# Page configuration
st.set_page_config(page_title="📢 Compte-rendu de réunion", page_icon=":microphone:", layout="centered")
if st.secrets["app"]["use_custom_style"]:
    st.markdown(custom_font(), unsafe_allow_html=True)
st.logo("./app/static/logo.png", size="large")

# CAS authentication
if not st.experimental_user.is_logged_in:
    st.button("🔑 Se connecter avec votre compte universitaire", on_click=st.login)
    st.stop()

st.button("🚪 Se déconnecter", on_click=st.logout)
st.markdown(f"👋 Bonjour {st.experimental_user.name}, prêt à transformer vos discussions en compte-rendu ?")

st.title("Compte-rendu de réunion")

selected_model = st.selectbox("Choisissez la précision de l'analyse :", WHISPER_MODEL_OPTIONS, index=1)
diarization_enabled = st.checkbox("🔍 Identifier les différents intervenants", value=False)


@st.cache_resource
def load_whisper_model(model_name: str) -> whisper.Whisper:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(model_name, device=device)


@st.cache_resource
def load_diarization_model() -> Pipeline:
    access_token = st.secrets["huggingface"]["token"]
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=access_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    return pipeline


model = load_whisper_model(selected_model)
diarization_model = load_diarization_model() if diarization_enabled else None


@st.cache_data
def transcribe_audio(file_path: str) -> dict:
    return model.transcribe(file_path, language=None)


@st.cache_data
def diarize_audio(file_path: str) -> Any:
    return diarization_model(file_path) if diarization_model is not None else None


@st.cache_data
def summarize(text: str, num_sentences: int, language: str) -> str:
    summarize = summarize_text(text, num_sentences=num_sentences, language=language)
    format = format_summary(
        st.secrets["llm"]["url"],
        st.secrets["llm"]["token"],
        st.secrets["llm"]["model"],
        st.secrets["llm"]["max_tokens"],
        summarize,
        language=language,
    )
    return format


# Choose input option (file upload or microphone)
input_option = st.radio(
    "Comment souhaitez-vous ajouter l'audio ?", ("📂 Télécharger un fichier", "🎤 Utiliser le micro")
)

# Keep track of the transcription result and summary in session state
if "transcription_result" not in st.session_state:
    st.session_state.transcription_result = None
    st.session_state.summary = None

# Reset the transcription result and summary if the input option changes
if "previous_input_option" in st.session_state and st.session_state.previous_input_option != input_option:
    st.session_state.transcription_result = None
    st.session_state.summary = None

st.session_state.previous_input_option = input_option


def process_transcription(tmp_filename: str) -> None:
    """
    Process the transcription and diarization of the audio
    """
    logger.info(f"Starting transcription process for file '{tmp_filename}'")
    resampled_audio = None
    try:
        st.write("⏳ Analyse en cours... Prenez un café ☕")

        # Transcription Whisper
        transcription = transcribe_audio(tmp_filename)

        # Diarisation Pyannote (si activée)
        if diarization_enabled:
            st.write("🔍 Identification des intervenants en cours...")
            # Conversion et resampling
            resampled_audio = tmp_filename.rsplit(".", 1)[0] + "_resampled.wav"
            convert_and_resample_audio(tmp_filename, resampled_audio)
            diarization = diarize_audio(resampled_audio)
            transcription = assign_speakers(transcription, diarization)

        st.session_state.transcription_result = transcription
        st.session_state.summary = None
        logger.info(f"Successfully transcribed file '{tmp_filename}'")

    except Exception as e:
        logger.error(f"Error during transcription/diarization for file '{tmp_filename}': {str(e)}")
        st.error(f"❌ Erreur pendant la transcription/diarisation : {str(e)}")
    finally:
        os.remove(tmp_filename)
        if diarization_enabled and resampled_audio is not None:
            os.remove(resampled_audio)


# File upload option
if input_option == "📂 Télécharger un fichier":
    uploaded_file = st.file_uploader("Déposez votre fichier audio ici", type=["mp3", "wav", "m4a"])

    if uploaded_file is not None:
        if st.button("📝 Transformer l'audio en texte"):
            file_extension = uploaded_file.name.split(".")[-1]
            mime_type, _ = mimetypes.guess_type(uploaded_file.name)
            if mime_type and mime_type.startswith("audio"):
                logger.info(f"User '{st.experimental_user.name}' uploaded file '{uploaded_file.name}' for transcription.")
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_filename = tmp_file.name
                process_transcription(tmp_filename)
            else:
                logger.warning(f"Uploaded file '{uploaded_file.name}' has unsupported format.")
                st.error("❌ Format non reconnu ! Merci d'ajouter un fichier audio valide.")

# Microphone input option
elif input_option == "🎤 Utiliser le micro":
    if "audio_data" not in st.session_state:
        st.session_state.audio_data = None
        st.write("🎤 Cliquez pour enregistrer votre réunion")

    audio_data = st.audio_input("Enregistrez votre message vocal")

    if audio_data:
        # Convert the audio data to bytes
        audio_bytes = audio_data.getvalue()
        # Keep the audio data in session state
        st.session_state.audio_data = audio_bytes

        st.write("✅ Enregistrement terminé.")

    # Add a button to start the transcription
    if "audio_data" in st.session_state and st.session_state.audio_data:
        if st.button("📝 Transformer l'audio en texte"):
            logger.info(f"User '{st.experimental_user.name}' is using microphone for transcription.")
            st.session_state.transcription_result = None
            st.session_state.summary = None
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(st.session_state.audio_data)
                tmp_filename = tmp_file.name
            process_transcription(tmp_filename)

# Show the transcription result
if "transcription_result" in st.session_state and st.session_state.transcription_result:
    result = st.session_state.transcription_result
    detected_language = result["language"]
    text_transcription = result["text"]

    st.write(f"🌍 Langue détectée : **{detected_language}**")

    transcript = "\n".join(f"{seg['text']}" for seg in st.session_state.transcription_result["segments"])

    st.write("🗒️ Voici votre compte-rendu brut :")
    st.code(transcript, language="plaintext", height=200, wrap_lines=True)
    json_content = json.dumps(result, indent=4)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📄 Télécharger la transcription (TXT)", transcript, "transcription.txt", "text/plain")
    with col2:
        st.download_button(
            "📝 Télécharger la version détaillée (JSON)", json_content, "transcription.json", "application/json"
        )

    if diarization_enabled:
        st.write("🗣️ Transcription avec intervenants")

        transcript_with_speakers = "\n".join(
            f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg.get('speaker', 'Speaker ?')}: {seg['text']}"
            for seg in st.session_state.transcription_result["segments"]
        )

        st.code(transcript_with_speakers, height=200, wrap_lines=True)

        st.download_button(
            "📑 Télécharger la version annotée avec les intervenants",
            transcript_with_speakers,
            "transcript_with_speakers.txt",
            "text/plain",
        )

    # Generate a summary of the transcription
    st.subheader("Synthèse du compte-rendu")

    if "summary" not in st.session_state:
        st.session_state.summary = None

    num_sentences = st.slider(
        "Choisissez le nombre de lignes pertinentes à extraire",
        min_value=5,
        max_value=300,
        value=st.secrets["app"].get("sumy_length_default", 80),
    )

    if st.button("✨ Générer une synthèse"):
        logger.info(f"User '{st.experimental_user.name}' is generating a summary.")
        try:
            st.write("⏳ Analyse en cours... Prenez un café ☕")
            summary = summarize(text_transcription, num_sentences=num_sentences, language=detected_language)
            st.session_state.summary = summary
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            st.error(str(e))

    if st.session_state.summary:
        st.code(st.session_state.summary, language="plaintext", height=200, wrap_lines=True)

        st.download_button(
            "📥 Télécharger la synthèse (Markdown)",
            st.session_state.summary,
            "synthese.md",
            "text/markdown",
        )
