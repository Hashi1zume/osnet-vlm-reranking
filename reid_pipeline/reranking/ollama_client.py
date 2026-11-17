import base64

import requests


class OllamaVLMClient:
    """Small Ollama adapter so the reranker does not depend on HTTP details."""

    def __init__(self, model: str = "qwen2.5vl:7b", endpoint: str = "http://127.0.0.1:11434/api/generate"):
        self.model = model
        self.endpoint = endpoint

    def generate(self, images: list[bytes], prompt: str) -> str:
        # The client returns raw model text; schema validation belongs to the reranker
        # so transport failures and model-quality failures remain distinguishable.
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [base64.b64encode(image).decode("ascii") for image in images],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 256},
        }
        response = requests.post(self.endpoint, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()["response"]
