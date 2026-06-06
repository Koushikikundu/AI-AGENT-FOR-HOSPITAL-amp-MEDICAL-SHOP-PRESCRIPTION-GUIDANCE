from __future__ import annotations

import requests

from backend.config import HUGGINGFACE_API_KEY, HUGGINGFACE_INFERENCE_MODEL, MAX_NEW_TOKENS


class HuggingFaceService:
    def __init__(
        self,
        api_key: str = HUGGINGFACE_API_KEY,
        model: str = HUGGINGFACE_INFERENCE_MODEL,
    ) -> None:
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("HUGGINGFACE_API_KEY is not configured.")

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{self.model}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "return_full_text": False,
                },
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            return str(data[0].get("generated_text", "")).strip()
        if isinstance(data, dict) and "generated_text" in data:
            return str(data["generated_text"]).strip()
        raise RuntimeError(f"Unexpected HuggingFace response: {data}")
