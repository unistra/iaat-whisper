import streamlit as st
from utils.style import custom_font
import os
from utils.log import setup_logger
from streamlit.runtime.secrets import secrets_singleton
from utils.secrets import get_secrets

# Setup logger
setup_logger()

# Secrets management
secrets_path = ".streamlit/secrets.toml"
if not os.path.exists(secrets_path):
    secrets_singleton._secrets = get_secrets()

st.set_page_config(page_title="IA de transcription", page_icon="📌", layout="centered")
if st.secrets["app"]["use_custom_style"]:
    st.markdown(custom_font(), unsafe_allow_html=True)
st.logo("./app/static/logo.png", size="large")

# CAS authentication
if not st.user.is_logged_in:
    st.button("🔑 Se connecter avec votre compte universitaire", on_click=st.login)
else:
    st.button("🚪 Se déconnecter", on_click=st.logout)
    st.markdown(f"👋 Bonjour {st.user.name}, prêt à utiliser l'IA de transcription ?")

st.title("IA de Transcription et Sous-Titrage")
st.write(
    "Bienvenue dans votre assistant intelligent pour transformer l'audio en texte et générer des sous-titres de vidéos."
)

st.markdown(
    "⚠️ **Proof of Concept expérimental** : Cette application est en phase de test préliminaire (ni alpha, ni beta)."
)

st.markdown("📱 **Compatibilité** : L'application est optimisée pour une utilisation sur mobile et desktop.")

st.markdown("📂 **Limite actuelle** : 10 Go par fichier audio.")

st.markdown("🔑 **Authentification** : Utilisation de cas, un compte est nécessaire.")

st.markdown(
    "💡 **Feedback bienvenu** : Si vous souhaitez tester et nous faire un retour, écrivez-nous à [dnum-ia@unistra.fr](mailto:dnum-ia@unistra.fr)."
)

st.header("Transcription")
st.write(
    "Téléversez un fichier audio ou utilisez le micro pour générer une transcription, identifiez les intervenants et créez un compte-rendu en Markdown."
)
st.page_link("pages/1_Transcription.py", label="Accéder à la transcription", icon="📝")

st.header("Sous-titrage de vidéos")
st.write("Téléversez votre vidéo, générez des sous-titres synchronisés et traduisez-les si nécessaire.")
st.page_link("pages/2_Sous-titrage.py", label="Accéder au sous-titrage", icon="🎬")
