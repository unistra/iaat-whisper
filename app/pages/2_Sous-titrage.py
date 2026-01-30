import streamlit as st
import tempfile
import mimetypes
import os

from utils.log import logger, setup_logger
from utils.secrets import get_secrets
from utils.style import custom_font
from streamlit.runtime.secrets import secrets_singleton
from utils.process import extract_audio_from_video, translate_srt_in_chunks
from utils.api import transcribe_audio_via_api
from utils.resource import get_gpu_lock, get_whisper_model

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
if not st.user.is_logged_in:
    st.button("🔑 Se connecter avec votre compte universitaire", on_click=st.login)
    st.stop()

st.button("🚪 Se déconnecter", on_click=st.logout)
st.markdown(f"👋 Bonjour {st.user.name}, prêt à générer des sous-titres ?")

st.title("Sous-titrage de vidéos")

lock = get_gpu_lock()
# st.write(f"Tâches disponibles : {lock._value} / {MAX_JOBS}")
# st.write(f"ID du verrou actuel : {id(lock)}")

WHISPER_LANGUAGES = {
    "Auto-détection": None,
    "Français": "fr",
    "Anglais": "en",
    "Espagnol": "es",
    "Allemand": "de",
    "Italien": "it"
}

if "selected_language_code_subtitling" not in st.session_state:
    st.session_state.selected_language_code_subtitling = None

selected_language_name = st.selectbox(
    "Forcer la langue de transcription (facultatif)",
    options=list(WHISPER_LANGUAGES.keys()),
    index=0, # Default to "Auto-détection"
    help="Sélectionnez une langue pour la transcription. L'auto-détection est utilisée par défaut."
)
st.session_state.selected_language_code_subtitling = WHISPER_LANGUAGES[selected_language_name]

@st.cache_data
def transcribe_audio(file_path: str, language: str | None) -> dict:
    if st.secrets["app"].get("transcription_mode", "local") == "api":
        return transcribe_audio_via_api(
            st.secrets["llm"]["url"],
            st.secrets["llm"]["token"],
            st.secrets["app"].get("whisper_model", "turbo"),
            file_path,
            timestamp_granularities=["segment", "word"],
            language=language,
        )
    else:
        with get_whisper_model() as model:
            return model.transcribe(file_path, language=language)

@st.cache_data
def translate(base_url: str, authtoken: str, model: str, max_tokens, srt_text: str, language: str = "en") -> str:
    return translate_srt_in_chunks(base_url, authtoken, model, max_tokens, srt_text, language, 20000)


# Gestion de l'état de la session
if "subtitle_result" not in st.session_state:
    st.session_state.subtitle_result = None


if "disabled_transcription" not in st.session_state:
    st.session_state.disabled_transcription = False

def on_click_transcription():
    st.session_state.disabled_transcription = True

if "disabled_translation" not in st.session_state:
    st.session_state.disabled_translation = False

def on_click_translation():
    st.session_state.disabled_translation = True

# Chargement du fichier vidéo
uploaded_video = st.file_uploader("Déposez votre fichier vidéo ici", type=["m4v", "mp4", "mov", "avi"], help="Formats supportés : m4v, mp4, mov, avi")

if uploaded_video is not None:
    if st.button("📝 Générer les sous-titres", disabled=st.session_state.disabled_transcription, on_click=on_click_transcription):
        logger.info(f"User '{st.user.name}' uploaded file '{uploaded_video.name}' for subtitling.")
        file_extension = uploaded_video.name.split(".")[-1]
        mime_type, _ = mimetypes.guess_type(uploaded_video.name)

        if mime_type and mime_type.startswith("video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name

            # Extraire l'audio de la vidéo
            audio_path = video_path.rsplit(".", 1)[0] + ".wav"
            extract_audio_from_video(video_path, audio_path)

            with st.spinner("⏳ En attente de disponibilité..."):
                acquired = lock.acquire(blocking=True, timeout=300)

            if not acquired:
                st.warning("⚠️ Le système est actuellement très sollicité. Merci de réessayer dans quelques instants.")
                logger.warning(f"GPU lock is currently held, user '{st.user.name}' must wait to process file '{video_path}'")
                st.stop()

            try:
                logger.info(f"Starting subtitle generation for file '{video_path}'")
                st.write("⏳ Analyse en cours... Prenez un café ☕")
                transcription = transcribe_audio(audio_path, st.session_state.selected_language_code_subtitling)

                # Stocker la transcription dans la session
                st.session_state.subtitle_result = transcription
                logger.info(f"Successfully generated subtitles for file '{video_path}'")

            except Exception as e:
                logger.error(f"Error during subtitle generation for file '{video_path}': {str(e)}")
                st.error(f"❌ Une erreur est survenue : {e}")

            finally:
                lock.release()
                os.remove(video_path)
                os.remove(audio_path)
                st.session_state.disabled_transcription = False

# Affichage des résultats
if st.session_state.subtitle_result:
    result = st.session_state.subtitle_result
    detected_language = result.get("language", "fr")
    st.write(f"🌍 Langue détectée : **{detected_language}**")

    # Génération des fichiers SRT et VTT
    srt_path = tempfile.mktemp(suffix=".srt")
    vtt_path = tempfile.mktemp(suffix=".vtt")
    try:
        from whisper.utils import get_writer

        writer_srt = get_writer("srt", os.path.dirname(srt_path))
        with open(srt_path, "w", encoding="utf-8") as f:
            writer_srt.write_result(result, file=f)  # type: ignore

        writer_vtt = get_writer("vtt", os.path.dirname(vtt_path))
        with open(vtt_path, "w", encoding="utf-8") as f:
            writer_vtt.write_result(result, file=f)  # type: ignore

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
        translate_enabled = st.checkbox("Traduire les sous-titres (expérimental)", value=False, help="Cochez cette case pour afficher les options de traduction des sous‑titres générés.")
        if translate_enabled:
            language_labels = {
                "fr": "French",
                "en": "English",
                "de": "German",
                "it": "Italian",
                "es": "Spanish",
            }
            language_labels_fr = {
                "fr": "Français",
                "en": "Anglais",
                "de": "Allemand",
                "it": "Italien",
                "es": "Espagnol",
            }
            language_target = st.selectbox(
                "Choisissez la langue de traduction :", filter(lambda x: x != detected_language, ["", "fr", "en", "de", "it", "es"]), index=0,
                format_func=lambda x: language_labels_fr[x] if x != "" else "",
            )

            if language_target != "" and language_target != detected_language and st.button("🌐 Traduire les sous-titres", disabled=st.session_state.disabled_translation, on_click=on_click_translation):
                logger.info(f"User '{st.user.name}' is translating subtitles to '{language_target}'.")
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
        raise e
    finally:
        if "srt_path" in locals() and os.path.exists(srt_path):
            os.remove(srt_path)
        if "vtt_path" in locals() and os.path.exists(vtt_path):
            os.remove(vtt_path)
        st.session_state.disabled_translation = False
