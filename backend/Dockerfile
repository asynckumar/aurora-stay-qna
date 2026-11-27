FROM python:3.13.15-slim

WORKDIR /app

# Install dependencies first so code changes do not invalidate this layer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
# CPU-only PyTorch, same version the code was tested with
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HOME=/app/huggingface

# Download the embedding model at build time so the container starts fast
RUN python -c "\
from langchain_huggingface import HuggingFaceEmbeddings; \
HuggingFaceEmbeddings( \
    model_name='HarishMaths/Hotel-Policy-Embedding', \
    model_kwargs={'device': 'cpu'}, \
    encode_kwargs={'normalize_embeddings': True} \
)"

# Force offline mode at runtime so it uses the cached model
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV PYTHONUNBUFFERED=1

# Feedback and question history go to /data, so a volume mounted there
# keeps them when the container is removed.
RUN mkdir -p /data
ENV FEEDBACK_PATH=/data/feedback.jsonl HISTORY_PATH=/data/history.db

COPY . .

EXPOSE 7860

# The model and index take a while to load, so wait 60 s before counting failures
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health', timeout=3)"

# One process (the model and index are loaded once) with 8 threads, so
# several staff members can be served at the same time, including streams.
CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "8", "--timeout", "120", "-b", "0.0.0.0:7860", "app:rag_api"]
