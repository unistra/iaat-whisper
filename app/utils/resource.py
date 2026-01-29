import streamlit as st
from typing import Any
import threading

MAX_JOBS = 1

@st.cache_resource
def load_whisper_model(model_name: str) -> Any:
    import torch
    import whisper

    if st.secrets["app"].get("transcription_mode", "local") == "local":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return whisper.load_model(model_name, device=device)
    return None

@st.cache_resource
def load_diarization_model() -> Any:
    import torch
    from pyannote.audio import Pipeline

    access_token = st.secrets["huggingface"]["token"]
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=access_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    return pipeline

@st.cache_resource
def get_gpu_lock():
    sem = threading.Semaphore(MAX_JOBS)
    return sem
