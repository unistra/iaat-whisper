FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg libportaudio2 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
COPY app ./app
COPY templates ./templates
COPY .streamlit/config.toml ./.streamlit/config.toml

RUN python3.11 -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "app/Accueil.py", "--server.port=8501", "--server.address=0.0.0.0"]
