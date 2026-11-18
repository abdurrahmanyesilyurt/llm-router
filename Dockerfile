FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LLM_ROUTER_EMBEDDING_CACHE_DIR=/models \
    LLM_ROUTER_KNN_EXAMPLES=/app/eval/prompts.jsonl

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install . \
 && useradd --create-home --uid 10001 app \
 && mkdir -p /models && chown app /models
COPY eval/prompts.jsonl ./eval/prompts.jsonl

USER app
EXPOSE 8000
# The embedding model is downloaded into the /models volume on first start.
CMD ["uvicorn", "llm_router.api:app", "--host", "0.0.0.0", "--port", "8000"]
