# Single image used for both the API and the UI (compose / Cloud Run pick the
# command). Slim base keeps the image small; the pure-Python wheels we depend
# on need no build toolchain.
FROM python:3.12-slim

# - PYTHONDONTWRITEBYTECODE: no .pyc files (smaller, cleaner container)
# - PYTHONUNBUFFERED: logs stream immediately (important for docker/Cloud logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (separate layer) so code changes don't bust the
# pip cache on every rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and Streamlit theme.
COPY app ./app
COPY .streamlit ./.streamlit
COPY scripts/brand_streamlit.py ./scripts/brand_streamlit.py

# Brand Streamlit's served HTML so shared links unfurl as MobiHelp instead of
# Streamlit. Link-preview crawlers read the static HTML before JS runs, so we
# bake the title, favicon and Open Graph/Twitter tags into index.html at build
# time. Only affects the UI service (the API doesn't serve this HTML).
ARG PUBLIC_BASE_URL=https://mobisystems.help
ENV PUBLIC_BASE_URL=${PUBLIC_BASE_URL}
RUN python scripts/brand_streamlit.py

# Cloud Run injects $PORT (usually 8080); locally we default to 8000. Shell
# form is required so ${PORT} is expanded at runtime. This is the API service;
# the UI service overrides the command (in compose / Cloud Run).
EXPOSE 8000
CMD uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
