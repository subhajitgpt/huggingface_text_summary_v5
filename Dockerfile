FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    TRANSFORMERS_CACHE=/app/.hf \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# System deps (minimal)
RUN apt-get update \
    ; apt-get install -y --no-install-recommends git \
    ; rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY app.py /app/app.py

RUN pip install --upgrade pip \
    ; pip install .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app.py"]
