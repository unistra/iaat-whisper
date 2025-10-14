FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y ffmpeg libportaudio2 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY .python-version ./
COPY pyproject.toml .
COPY uv.lock .
COPY app ./app
COPY templates ./templates
COPY .streamlit/config.toml ./.streamlit/config.toml

RUN uv sync --locked
ENV PATH=".venv/bin:$PATH"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["uv", "run", "streamlit", "run", "app/Accueil.py", "--server.port=8501", "--server.address=0.0.0.0"]
