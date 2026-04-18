FROM paddlepaddle/paddle:3.3.0-gpu-cuda13.0-cudnn9.13

WORKDIR /app

RUN apt-get update

RUN pip install --no-cache-dir \
    paddleocr \
    fastapi \
    uvicorn \
    python-multipart

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
