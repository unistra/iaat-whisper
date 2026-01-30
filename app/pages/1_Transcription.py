import streamlit as st
import tempfile
import json
import os
import mimetypes
import io

from markdown_pdf import MarkdownPdf, Section
from utils.log import logger, setup_logger
from streamlit.runtime.secrets import secrets_singleton
from typing import Any
from utils.style import custom_font
from utils.api import format_summary, transcribe_audio_via_api
from utils.secrets import get_secrets
from utils.process import (
    download_nltk_resources,
    convert_and_resample_audio,
    extract_audio_from_video,
    summarize_text,
    assign_speakers,
)
from utils.resource import get_whisper_model, load_diarization_model, get_gpu_lock

# FIX Docker + m4a
mimetypes.init()
mimetypes.add_type("audio/mp4", ".m4a")
mimetypes.add_type("video/mp4", ".m4v")

# Setup logger
setup_logger()

# Secrets management
secrets_path = ".streamlit/secrets.toml"
if not os.path.exists(secrets_path):
    secrets_singleton._secrets = get_secrets()

# Page configuration
st.set_page_config(page_title="📢 Transcription", page_icon=":microphone:", layout="centered")

if st.secrets["app"]["use_custom_style"]:
    st.markdown(custom_font(), unsafe_allow_html=True)
st.logo("./app/static/logo.png", size="large")

# CAS authentication
if not st.user.is_logged_in:
    st.button("🔑 Se connecter avec votre compte universitaire", on_click=st.login)
    st.stop()

st.button("🚪 Se déconnecter", on_click=st.logout)
st.markdown(f"👋 Bonjour {st.user.name}, prêt à transformer vos discussions en compte-rendu ?")

st.title("Transcription")

lock = get_gpu_lock()
# st.write(f"Tâches disponibles : {lock._value} / {MAX_JOBS}")
# st.write(f"ID du verrou actuel : {id(lock)}")

def on_diarization_change():
    st.session_state.transcription_result = None
    st.session_state.summary = None


diarization_enabled = st.checkbox(
    "🔍 Identifier les différents intervenants (expérimental)",
    value=False,
    on_change=on_diarization_change,
    help="Cochez cette case pour tenter d’identifier qui parle et à quel moment. Le traitement sera plus long et plus exigeant en ressources."
)

WHISPER_LANGUAGES = {
    "Auto-détection": None,
    "Français": "fr",
    "Anglais": "en",
    "Espagnol": "es",
    "Allemand": "de",
    "Italien": "it"
}

if "selected_language_code" not in st.session_state:
    st.session_state.selected_language_code = None

selected_language_name = st.selectbox(
    "Forcer la langue de transcription (facultatif)",
    options=list(WHISPER_LANGUAGES.keys()),
    index=0, # Default to "Auto-détection"
    on_change=on_diarization_change,
    help="Sélectionnez une langue pour la transcription. L'auto-détection est utilisée par défaut."
)
st.session_state.selected_language_code = WHISPER_LANGUAGES[selected_language_name]

diarization_model = load_diarization_model() if diarization_enabled else None


@st.cache_data
def transcribe_audio(file_path: str, language: str | None) -> dict:
    if st.secrets["app"].get("transcription_mode", "local") == "api":
        return transcribe_audio_via_api(
            st.secrets["llm"]["url"],
            st.secrets["llm"]["token"],
            st.secrets["app"].get("whisper_model", "turbo"),
            file_path,
            language=language,
        )
    else:
        with get_whisper_model() as model:
            return model.transcribe(file_path, language=language)

@st.cache_data
def diarize_audio(file_path: str) -> Any:
    return diarization_model(file_path) if diarization_model is not None else None


@st.cache_data
def summarize(
    text: str,
    num_sentences: int,
    language: str,
    prompt_template: str | None,
    custom_prompt: str | None = None,
) -> str:
    summarize_text_result = summarize_text(text, num_sentences=num_sentences, language=language)

    kwargs = {
        "base_url": st.secrets["llm"]["url"],
        "authtoken": st.secrets["llm"]["token"],
        "model": st.secrets["llm"]["model"],
        "max_tokens": st.secrets["llm"]["max_tokens"],
        "temperature": st.secrets["llm"]["temperature"],
        "summary": summarize_text_result,
        "language": language,
    }

    if custom_prompt:
        kwargs["custom_prompt"] = custom_prompt
    elif prompt_template:
        kwargs["prompt_template"] = prompt_template

    return format_summary(**kwargs)


# Choose input option (file upload or microphone)
input_option = st.radio(
    "Comment souhaitez-vous ajouter l'audio ?", ("📂 Téléverser un fichier", "🎤 Utiliser le micro")
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

if "disabled_transcription" not in st.session_state:
    st.session_state.disabled_transcription = False

def on_click_transcription():
    st.session_state.disabled_transcription = True
    st.session_state.transcription_result = None
    st.session_state.summary = None

if "disabled_summarize" not in st.session_state:
    st.session_state.disabled_summarize = False

def on_click_summarize():
    st.session_state.disabled_summarize = True

def process_transcription(tmp_filename: str, type: str = "audio", language: str | None = None) -> None:
    """
    Process the transcription and diarization of the audio
    """
    logger.info(f"Starting transcription process for file '{tmp_filename}'")
    resampled_audio = None
    audio_file = tmp_filename

    with st.spinner("⏳ En attente de disponibilité..."):
        acquired = lock.acquire(blocking=True, timeout=300)

    if not acquired:
        st.warning("⚠️ Le système est actuellement très sollicité. Merci de réessayer dans quelques instants.")
        logger.warning(f"GPU lock is currently held, user '{st.user.name}' must wait to process file '{tmp_filename}'")
        st.stop()

    try:
        st.write("⏳ Analyse en cours... Prenez un café ☕")

        if type == "video":
            logger.info(f"Extracting audio from video file '{tmp_filename}'")
            # Extraire l'audio de la vidéo
            audio_file = tmp_filename.rsplit(".", 1)[0] + ".wav"
            extract_audio_from_video(tmp_filename, audio_file)

        # Transcription
        transcription = transcribe_audio(audio_file, language)

        # Diarisation Pyannote (si activée)
        if diarization_enabled:
            st.write("🔍 Identification des intervenants en cours...")
            # Conversion et resampling
            resampled_audio = audio_file.rsplit(".", 1)[0] + "_resampled.wav"
            convert_and_resample_audio(audio_file, resampled_audio)
            diarization = diarize_audio(resampled_audio)
            transcription = assign_speakers(transcription, diarization)

        st.session_state.transcription_result = transcription
        st.session_state.summary = None
        logger.info(f"Successfully transcribed file '{audio_file}'")

    except Exception as e:
        logger.error(f"Error during transcription/diarization for file '{audio_file}': {str(e)}")
        st.error(f"❌ Erreur pendant la transcription/diarisation : {str(e)}")
    finally:
        lock.release()
        os.remove(tmp_filename)
        if audio_file != tmp_filename:
            os.remove(audio_file)
        if diarization_enabled and resampled_audio is not None:
            os.remove(resampled_audio)
        st.session_state.disabled_transcription = False


# File upload option
if input_option == "📂 Téléverser un fichier":
    uploaded_file = st.file_uploader("Déposez votre fichier audio ici", type=["mp3", "wav", "m4a", "m4v", "mp4", "mov", "avi"], help="Formats supportés : mp3, wav, m4a, m4v, mp4, mov, avi")

    if uploaded_file is not None:
        if st.button("📝 Transformer l'audio en texte", disabled=st.session_state.disabled_transcription, on_click=on_click_transcription):
            file_extension = uploaded_file.name.split(".")[-1]
            mime_type, _ = mimetypes.guess_type(uploaded_file.name)
            if mime_type and (mime_type.startswith("audio") or  mime_type.startswith("video")):
                logger.info(f"User '{st.user.name}' uploaded file '{uploaded_file.name}' for transcription.")
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_filename = tmp_file.name
                process_transcription(tmp_filename, "video" if mime_type.startswith("video") else "audio", st.session_state.selected_language_code)
            else:
                logger.warning(f"Uploaded file '{uploaded_file.name}' has unsupported format : {mime_type}")
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
        if st.button("📝 Transformer l'audio en texte", disabled=st.session_state.disabled_transcription, on_click=on_click_transcription):
            logger.info(f"User '{st.user.name}' is using microphone for transcription.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(st.session_state.audio_data)
                tmp_filename = tmp_file.name
            process_transcription(tmp_filename, language=st.session_state.selected_language_code)

# Show the transcription result
if "transcription_result" in st.session_state and st.session_state.transcription_result:
    result = st.session_state.transcription_result
    detected_language = result["language"]
    text_transcription = result["text"]

    st.write(f"🌍 Langue détectée : **{detected_language}**")

    transcript = "\n".join(f"{seg['text']}" for seg in st.session_state.transcription_result["segments"])

    st.write("🗒️ Voici votre transcription :")
    st.code(transcript, language="plaintext", height=200, wrap_lines=True)
    json_content = json.dumps(result, indent=4)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📄 Télécharger la transcription (TXT)", transcript, "transcription.txt", "text/plain")
    with col2:
        st.download_button(
            "📝 Télécharger la version détaillée (JSON)", json_content, "transcription.json", "application/json"
        )

    transcript_with_speakers = ""
    if diarization_enabled:
        st.write("🗣️ Transcription avec intervenants")

        transcript_with_speakers = "\n".join(
            f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg.get('speaker', 'Speaker ?')}: {seg['text']}"
            for seg in st.session_state.transcription_result["segments"]
        )

        st.code(transcript_with_speakers, height=200, wrap_lines=True)

        st.download_button(
            "📑 Télécharger la version annotée (TXT)",
            transcript_with_speakers,
            "transcript_with_speakers.txt",
            "text/plain",
        )

    # Generate a summary of the transcription
    summarize_enabled = st.checkbox("Créer un compte-rendu (expérimental)", value=False, help="Cochez cette case pour afficher les options de génération de synthèse du texte transcrit.")
    if summarize_enabled:
        st.subheader("Compte-rendu")

        # Define prompt choices
        PROMPT_CHOICES = {
            "Compte-rendu de réunion": "meeting_report_prompt.j2",
            "Résumé de présentation": "presentation_summary_prompt.j2",
            "Synthèse de discussion": "discussion_summary_prompt.j2",
            "Prise de note rapide": "brainstorming_summary_prompt.j2",
            "Interview (Q&A)": "interview_summary_prompt.j2",
        }

        use_custom_prompt = st.checkbox("Utiliser un prompt personnalisé", help="Cochez cette case pour saisir votre propre prompt d’instructions afin de générer la synthèse à la place des options prédéfinies.")
        custom_prompt_text = None
        prompt_choice = None

        if use_custom_prompt:
            custom_prompt_text = st.text_area("Saisissez votre prompt personnalisé ici :", height=150)
        else:
            prompt_choice = st.selectbox(
                "Choisissez le type de synthèse :",
                options=list(PROMPT_CHOICES.keys()),
            )

        if "summary" not in st.session_state:
            st.session_state.summary = None

        num_sentences = st.slider(
            "Choisissez le nombre de lignes pertinentes à extraire",
            min_value=5,
            max_value=300,
            value=st.secrets["app"].get("sumy_length_default", 80),
            help="Choisir un nombre plus élevé donnera une synthèse plus longue et plus détaillée.",
        )

        if st.button("✨ Générer une synthèse", disabled=st.session_state.disabled_summarize, on_click=on_click_summarize):
            logger.info(f"User '{st.user.name}' is generating a summary.")
            try:
                st.write("⏳ Analyse en cours... Prenez un café ☕")

                download_nltk_resources()

                text_to_summarize = text_transcription
                if diarization_enabled:
                    text_to_summarize = transcript_with_speakers

                selected_template = None
                if not use_custom_prompt and prompt_choice:
                    selected_template = PROMPT_CHOICES[prompt_choice]

                if use_custom_prompt and not custom_prompt_text:
                    st.error("Le prompt personnalisé ne peut pas être vide.")
                    st.stop()

                summary = summarize(
                    text_to_summarize,
                    num_sentences=num_sentences,
                    language=detected_language,
                    prompt_template=selected_template,
                    custom_prompt=custom_prompt_text,
                )
                st.session_state.summary = summary
            except Exception as e:
                logger.error(f"Error during summarization: {str(e)}")
                st.error(str(e))
            finally:
                st.session_state.disabled_summarize = False

        if st.session_state.summary:
            st.code(st.session_state.summary, language="plaintext", height=200, wrap_lines=True)

            col1_summary, col2_summary = st.columns(2)
            with col1_summary:
                st.download_button(
                "📥 Télécharger la synthèse (Markdown)",
                st.session_state.summary,
                "synthese.md",
                "text/markdown",
            )
            with col2_summary:
                # Build the PDF
                pdf = MarkdownPdf()
                pdf.meta["title"] = "Synthèse de la transcription"
                pdf.meta["author"] = st.user.name
                pdf.add_section(Section(st.session_state.summary, toc=False))
                # Download button
                pdf_bytes = io.BytesIO()
                pdf.save_bytes(pdf_bytes)
                st.download_button(
                    "📄 Télécharger la synthèse (PDF)",
                    pdf_bytes,
                    "synthese.pdf",
                    "application/pdf",
                )