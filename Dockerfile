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

# Cloud Run injects $PORT (usually 8080); locally we default to 8000. Shell
# form is required so ${PORT} is expanded at runtime. This is the API service;
# the UI service overrides the command (in compose / Cloud Run).
EXPOSE 8000
CMD uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
