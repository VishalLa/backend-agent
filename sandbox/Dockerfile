FROM python:3.12-slim

RUN groupadd -r sandbox && useradd -r -g sandbox -d /home/sandbox -m sandbox

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache

COPY server.py .
RUN mkdir -p /workspace && chown -R sandbox:sandbox /workspace /app

USER sandbox
EXPOSE 8787

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8787", "--workers", "1"]
