# Supply Chain Late Delivery Risk — Streamlit dashboard image.
# Serves the 4-tab dashboard against the committed 1K sample and pre-computed
# full-dataset metrics. The full 180K CSV is never baked in (see .dockerignore).

FROM python:3.12-slim

# libgomp1 is the OpenMP runtime XGBoost links against at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Streamlit defaults that suit a container: no usage stats, listen on all
# interfaces, and a writable config home for the non-root user.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HOME=/home/app

# Install dependencies first so code changes don't bust the pip layer.
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Non-root runtime user.
RUN useradd --create-home --uid 1000 app
COPY . .
RUN chown -R app:app /app
USER app

EXPOSE 8501

# Streamlit exposes a liveness endpoint at /_stcore/health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').read()==b'ok' else 1)"

CMD ["streamlit", "run", "app/streamlit_app.py"]
