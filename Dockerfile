FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# CPU-only torch first: the default PyPI wheel bundles CUDA and is several GB.
# Its version must match the torch pin in requirements.txt. The base image's
# bundled pip is too old to read the PyTorch index metadata, so upgrade it first.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so containers start without a
# download and can run offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p data/raw data/raw_docs data/processed \
    && chown -R appuser:appuser /app /opt/hf-cache
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f\"http://localhost:{os.environ.get('PORT', '8501')}/_stcore/health\")"

# PORT lets the same image serve the admin app on 8502 (see docker-compose.yml).
ENV PORT=8501 APP_FILE=app.py
CMD ["sh", "-c", "streamlit run \"$APP_FILE\" --server.port=\"$PORT\" --server.address=0.0.0.0 --server.headless=true"]
