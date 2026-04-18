import tomllib
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SimpleEmbedding:
    def __init__(self, vllm_url: str, model_name: str):
        self.vllm_url = vllm_url
        self.model_name = model_name

    def get_embedding(self, text: str) -> np.ndarray:
        task = "Determine if the following text is semantically similar to another text"
        formatted_text = f"Instruct: {task}\nQuery: {text}"

        payload = {
            "input": formatted_text,
            "model": self.model_name
        }

        response = requests.post(self.vllm_url, json=payload)
        response.raise_for_status()

        embedding = response.json()["data"][0]["embedding"]
        return np.array(embedding)
