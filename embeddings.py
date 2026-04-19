import tomllib
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SimpleEmbedding:
    def __init__(self, vllm_url: str, model_name: str):
        self.vllm_url = vllm_url
        self.model_name = model_name

    def get_embedding(self, text: str) -> np.ndarray:
        response = requests.post(self.vllm_url, params={"text": text})
        response.raise_for_status()

        embedding = response.json()
        return np.array(embedding)
