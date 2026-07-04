FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY shadowisland_tool ./shadowisland_tool
COPY model_bundle ./model_bundle
COPY scripts ./scripts

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[web]" \
    && python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    && python scripts/check_model_bundle.py \
    && mkdir -p jobs

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=3).read()"

CMD ["shadowisland", "serve", "--host", "0.0.0.0", "--port", "8765"]

