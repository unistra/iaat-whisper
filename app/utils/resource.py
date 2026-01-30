import streamlit as st
from typing import Any
import threading
import queue
from contextlib import contextmanager

# Maximum concurrent jobs
MAX_JOBS = st.secrets["app"].get("max_concurrent_jobs", 3)

@st.cache_resource
def load_whisper_model(model_name: str, instance: int) -> Any:
    import torch
    import whisper

    if st.secrets["app"].get("transcription_mode", "local") == "local":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return whisper.load_model(model_name, device=device)
    return None

@st.cache_resource
def init_whisper_pool(model_name: str) -> queue.Queue[Any]:
    """
    Initialize a pool of Whisper models for concurrent transcription
    """
    model_queue: queue.Queue[Any] = queue.Queue(maxsize=MAX_JOBS)
    for i in range(MAX_JOBS):
        model = load_whisper_model(model_name, i)
        model_queue.put(model)
    return model_queue

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

@contextmanager
def get_whisper_model(timeout: int = 30):
    """
    Context manager to safely get and release a Whisper model from the pool.
    
    Args:
        timeout: Maximum time to wait for a model (seconds)
        
    Yields:
        The Whisper model
        
    Raises:
        ValueError: If no model is available within timeout
    """
    model_name = st.secrets["app"].get("whisper_model", "turbo")
    model_pool = init_whisper_pool(model_name)
    
    try:
        model = model_pool.get(block=True, timeout=timeout)
    except queue.Empty:
        raise ValueError("All transcription models are currently busy. Please try again later.")
    
    try:
        yield model
    finally:
        model_pool.put(model)