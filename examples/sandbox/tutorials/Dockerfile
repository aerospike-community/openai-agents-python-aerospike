FROM python:3.14-slim@sha256:0514bc76712d64f8bb48adcb95b1ab0d88c69fee95d62fe4c18ee948bee9b1e1
COPY --from=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a /uv /bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        poppler-utils \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --system --no-cache-dir --index-strategy first-index --exclude-newer "7 days" pypdf

WORKDIR /workspace
