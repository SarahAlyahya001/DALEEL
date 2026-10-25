# DALEEL web application image: Waitress serves the site on port 8000.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    DALEEL_HOST=0.0.0.0 \
    DALEEL_PORT=8000

# OpenCV needs these system libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch builds (same versions as requirements.txt) keep the image small;
# their dependencies still come from PyPI.
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.3.1+cpu torchvision==0.18.1+cpu \
    && pip install -r requirements.txt

COPY . .

# Download and verify the pinned model files (saved owner-only, so make them readable),
# then run as an unprivileged user.
RUN python scripts/setup_weights.py \
    && chmod -R a+rX weights \
    && useradd --system --uid 10001 --home-dir /app daleel \
    && mkdir -p instance \
    && chown daleel:daleel instance

USER daleel
VOLUME ["/app/instance"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/login', timeout=4)"

CMD ["python", "run.py"]
