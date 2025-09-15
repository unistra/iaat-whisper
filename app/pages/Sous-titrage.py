import streamlit as st
from utils.log import logger, setup_logger
import whisper
import tempfile
import mimetypes
import os
from whisper.utils import get_writer
from utils.secrets import get_secrets
from utils.style import custom_font
from streamlit.runtime.secrets import secrets_singleton
from utils.process import WHISPER_MODEL_OPTIONS, extract_audio_from_video, translate_srt_in_chunks
import torch

# Setup logger
setup_logger()

# Gestion des secrets
secrets_path = ".streamlit/secrets.toml"
if not os.path.exists(secrets_path):
    secrets_singleton._secrets = get_secrets()

st.set_page_config(page_title="🎬 Sous-titrage de vidéos", page_icon=":film_strip:", layout="centered")
if st.secrets["app"]["use_custom_style"]:
    st.markdown(custom_font(), unsafe_allow_html=True)
st.logo("./app/static/logo.png", size="large")

# Authentification CAS
if not st.experimental_user.is_logged_in:
    st.button("🔑 Se connecter avec votre compte universitaire", on_click=st.login)
    st.stop()

st.button("🚪 Se déconnecter", on_click=st.logout)
st.markdown(f"👋 Bonjour {st.experimental_user.name}, prêt à générer des sous-titres ?")

st.title("Sous-titrage de vidéos")

# Sélection du modèle
selected_model = st.selectbox("Choisissez la précision de l'analyse :", WHISPER_MODEL_OPTIONS, index=1)


@st.cache_resource
def load_whisper_model(model_name: str) -> whisper.Whisper:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(model_name, device=device)


model = load_whisper_model(selected_model)

translate_option = st.checkbox("🌎 Traduire directement en anglais")


@st.cache_data
def transcribe_audio(file_path: str, translate: bool = False) -> dict:
    task = "translate" if translate else "transcribe"
    language = "en" if translate else None
    word_timestamps = True if translate else False
    return model.transcribe(file_path, language=language, task=task, word_timestamps=word_timestamps)


@st.cache_data
def translate(base_url: str, authtoken: str, model: str, max_tokens, srt_text: str, language: str = "en") -> str:
    return translate_srt_in_chunks(base_url, authtoken, model, max_tokens, srt_text, language, 20000)


# Gestion de l'état de la session
if "subtitle_result" not in st.session_state:
    st.session_state.subtitle_result = None

# Chargement du fichier vidéo
uploaded_video = st.file_uploader("Déposez votre fichier vidéo ici", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    if st.button("📝 Générer les sous-titres"):
        logger.info(f"User '{st.experimental_user.name}' uploaded file '{uploaded_video.name}' for subtitling.")
        file_extension = uploaded_video.name.split(".")[-1]
        mime_type, _ = mimetypes.guess_type(uploaded_video.name)

        if mime_type and mime_type.startswith("video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name

            # Extraire l'audio de la vidéo
            audio_path = video_path.rsplit(".", 1)[0] + ".wav"
            extract_audio_from_video(video_path, audio_path)

            try:
                logger.info(f"Starting subtitle generation for file '{video_path}'")
                st.write("⏳ Analyse en cours... Prenez un café ☕")
                transcription = transcribe_audio(audio_path, translate=translate_option)

                # Stocker la transcription dans la session
                st.session_state.subtitle_result = transcription
                logger.info(f"Successfully generated subtitles for file '{video_path}'")

            except Exception as e:
                logger.error(f"Error during subtitle generation for file '{video_path}': {str(e)}")
                st.error(f"❌ Une erreur est survenue : {e}")

            finally:
                os.remove(video_path)
                os.remove(audio_path)

# Affichage des résultats
if st.session_state.subtitle_result:
    result = st.session_state.subtitle_result
    detected_language = result["language"]
    st.write(f"🌍 Langue détectée : **{detected_language}**")

    # Génération des fichiers SRT et VTT
    srt_path = tempfile.mktemp(suffix=".srt")
    vtt_path = tempfile.mktemp(suffix=".vtt")
    try:
        writer_srt = get_writer("srt", os.path.dirname(srt_path))
        writer_srt(
            result,
            os.path.basename(srt_path),  # type: ignore
            {"max_line_width": 50, "max_line_count": 1, "highlight_words": False},
        )

        writer_vtt = get_writer("vtt", os.path.dirname(vtt_path))
        writer_vtt(
            result,
            os.path.basename(vtt_path),  # type: ignore
            {"max_line_width": 50, "max_line_count": 1, "highlight_words": False},
        )

        with open(srt_path, "r") as f:
            srt_content = f.read()

        with open(vtt_path, "r") as f:
            vtt_content = f.read()

        st.subheader("Sous-titres générés")
        st.code(srt_content, language="plaintext", height=200)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Télécharger les sous-titres (SRT)", srt_content, "subtitles.srt", "text/plain")
        with col2:
            st.download_button("📥 Télécharger les sous-titres (VTT)", vtt_content, "subtitles.vtt", "text/vtt")

        # Traduction du texte
        language_labels = {
            "fr": "French",
            "en": "English",
            "de": "German",
            "it": "Italian",
            "es": "Spanish",
        }
        language_target = st.selectbox(
            "Choisissez la langue de traduction :", ["", "fr", "en", "de", "it", "es"], index=0
        )

        if language_target != "" and language_target != detected_language:
            logger.info(f"User '{st.experimental_user.name}' is translating subtitles to '{language_target}'.")
            translated_text = translate(
                st.secrets["llm"]["url"],
                st.secrets["llm"]["token"],
                st.secrets["llm"]["model"],
                st.secrets["llm"]["max_tokens"],
                srt_content,
                language_labels[language_target],
            )

            # Affichage des sous-titres traduits
            st.subheader("📜 Sous-titres traduits")
            st.code(translated_text, language="plaintext", height=200)

            # Boutons de téléchargement des sous-titres traduits
            st.download_button(
                "💽 Télécharger les sous-titres traduits (SRT)",
                translated_text,
                "translated_subtitles.srt",
                "text/plain",
            )

    except Exception as e:
        logger.error(f"Error during subtitle file generation or translation: {str(e)}")
        st.error(f"❌ Une erreur est survenue : {e}")
    finally:
        os.remove(srt_path)
        os.remove(vtt_path)
