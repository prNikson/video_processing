FROM python:3.10-slim

RUN pip install --no-cache-dir torch transformers accelerate bitsandbytes fastapi uvicorn

WORKDIR /app

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8888"]
