from __future__ import annotations

import requests

from backend.config import MAX_NEW_TOKENS, OLLAMA_BASE_URL, OLLAMA_MODEL


class OllamaService:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": MAX_NEW_TOKENS},
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", "")).strip()
