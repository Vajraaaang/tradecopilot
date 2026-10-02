FROM ghcr.io/astral-sh/uv:0.11.3 AS uv

FROM python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable \
    && groupadd --gid 10001 forecast \
    && useradd --uid 10001 --gid forecast --create-home forecast \
    && mkdir /data \
    && chown forecast:forecast /data
USER 10001:10001
VOLUME ["/data"]
EXPOSE 8766
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import json, urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8766/health', timeout=3))['ok'] is True"]
# The demo command verifies and reuses an existing immutable synthetic report.
CMD ["sh", "-c", "tradecopilot forecast demo --output-dir /data/demo && exec tradecopilot forecast serve /data/demo/run/report.json --host 0.0.0.0 --port 8766"]
