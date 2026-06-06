from __future__ import annotations

import requests

from backend.config import GROQ_API_KEY, GROQ_MODEL, MAX_NEW_TOKENS, TEMPERATURE


class GroqService:
    def __init__(self, api_key: str = GROQ_API_KEY, model: str = GROQ_MODEL) -> None:
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a careful hospital medicine assistant."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_NEW_TOKENS,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"]).strip()
