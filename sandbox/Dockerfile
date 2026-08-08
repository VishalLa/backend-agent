FROM python:3.10-slim

# Install pandas, matplotlib, and server dependencies
RUN pip install --no-cache-dir \
    pandas \
    matplotlib \
    fastapi \
    uvicorn \
    pydantic

COPY worker.py /app/worker.py
WORKDIR /app

# Run Uvicorn listening strictly on a Unix socket, NOT a network port
CMD ["uvicorn", "worker:sandbox", "--uds", "/workspace/sandbox.sock"]