FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --requirement requirements.txt

RUN addgroup --system parseanything \
    && adduser --system --ingroup parseanything --no-create-home --home /app parseanything

COPY --chown=parseanything:parseanything app.py ./
COPY --chown=parseanything:parseanything backend ./backend
COPY --chown=parseanything:parseanything frontend ./frontend

USER parseanything

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=3)"]

CMD ["python", "app.py"]
