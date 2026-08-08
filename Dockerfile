# One image, two roles. The API and the Celery worker run the same code with a different
# command, so a worker can never be running a build the API has not seen -- the failure mode
# where a queue silently processes tasks with stale logic.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Requirements first so a code change does not reinstall the dependency tree.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY debate ./debate
COPY evals ./evals

# Runs as a non-root user. A forecasting service has no reason to hold root, and the
# container is the only place that is enforced.
RUN useradd --create-home --uid 10001 debate && chown -R debate:debate /app
USER debate

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
    sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status \
    == 200 else 1)"

CMD ["uvicorn", "debate.api:app", "--host", "0.0.0.0", "--port", "8000"]
