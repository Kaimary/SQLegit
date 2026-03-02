FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Prevent import-time crash when `.env` is not baked into the image.
    # Users can still override via `--env-file` / `-e`.
    TEST_INSTANCE_ROOT_PATH=/checklist/output/test_cases

WORKDIR /checklist

# System deps:
# - build-essential: some Python deps may compile native extensions depending on platform/wheels
# - git/curl/ca-certificates: common utilities (also helps debugging in container)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps pinned to the versions available in the current dev/runtime environment
# (Python 3.10.12). Keep this list minimal and reproducible.
RUN python -m pip install --upgrade pip \
    && python -m pip install \
      tqdm==4.67.1 \
      pandas==2.3.0 \
      numpy==2.2.6 \
      munch==4.0.0 \
      python-dotenv==1.1.0 \
      langchain==0.3.25 \
      langchain-core==0.3.74 \
      langchain-openai==0.3.31 \
      langchain-deepseek==0.1.4 \
      langchain-chroma==0.2.4 \
      chromadb==1.0.12 \
      sqlglot==26.26.0 \
      sql-metadata==2.17.0 \
      sqlparse==0.5.3 \
      datasketch==1.6.5 \
      networkx==3.4.2 \
      nltk==3.9.1 \
      func-timeout==4.3.5 \
      pydantic==2.11.5 \
      jinja2==3.1.6 \
      openai==2.15.0

# Copy only what we need to run (avoid baking secrets like `.env`, and keep image small).
COPY src/ src/
COPY templates/ templates/
COPY scripts/ scripts/
COPY run_judgment.py run_consensus.py evalution.py ./

# Runtime directories (logs, cached test cases, etc.)
RUN mkdir -p output/logs output/test_cases

ENTRYPOINT ["python"]
CMD ["run_judgment.py", "--help"]
