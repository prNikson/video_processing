from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

app = FastAPI()

model_name = "Qwen/Qwen3-Embedding-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModel.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float32
)

@app.post("/embeddings")
def embed(text: str) -> None:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

        return embedding
