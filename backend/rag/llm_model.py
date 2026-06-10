from __future__ import annotations

import logging

import requests

from backend.config import MAX_NEW_TOKENS, OLLAMA_BASE_URL, OLLAMA_MODEL, TEMPERATURE
from backend.services.groq_service import GroqService

logger = logging.getLogger(__name__)


class HospitalAnswerGenerator:
    def __init__(self, model_name: str = OLLAMA_MODEL) -> None:
        self.model_name = model_name
        self.ollama_model = model_name
        self.fallback_model = "google/flan-t5-base"
        self.groq_service = GroqService()
        self.task = "ollama"
        self.provider_used = "Ollama"
        self.loaded_model_name = self.ollama_model
        self.fallback_used = False
        self.secondary_fallback_used = False
        self.generator = None

        logger.warning("Answer model primary configured: %s", self.loaded_model_name)
        print(f"[HospitalAnswerGenerator] primary provider configured: Ollama ({self.loaded_model_name})")

    def _ensure_flan_generator(self) -> bool:
        if self.generator is not None:
            return True
        try:
            from transformers import pipeline

            self.generator = pipeline("text2text-generation", model=self.fallback_model)
            logger.warning("Answer model fallback loaded: %s", self.fallback_model)
            print(f"[HospitalAnswerGenerator] fallback model loaded: {self.fallback_model}")
            return True
        except Exception as exc:
            self.generator = None
            logger.warning("Answer model fallback failed: %s", exc)
            print(f"[HospitalAnswerGenerator] fallback model failed: {exc}")
            return False

    def get_runtime_info(self) -> dict:
        return {
            "requested_model": self.model_name,
            "loaded_model": self.loaded_model_name,
            "task": self.task,
            "fallback_used": self.fallback_used,
            "secondary_fallback_used": self.secondary_fallback_used,
            "provider_used": self.provider_used,
            "fallback_status": (
                "none"
                if not self.fallback_used and not self.secondary_fallback_used
                else "primary_failed_then_fallback"
                if self.fallback_used and not self.secondary_fallback_used
                else "all_fallbacks_used"
            ),
            "secondary_fallback_model": self.groq_service.model,
        }

    def _ollama_generate(self, prompt: str) -> str:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": TEMPERATURE,
                    "num_predict": MAX_NEW_TOKENS,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("response", "")).strip()

    def generate_prompt(self, prompt: str) -> str:
        print(f"[HospitalAnswerGenerator] generating with: {self.provider_used} ({self.loaded_model_name})")

        self.provider_used = "Ollama"
        self.task = "ollama"
        self.loaded_model_name = self.ollama_model

        if self.task == "ollama":
            try:
                text = self._ollama_generate(prompt)
                self.provider_used = "Ollama"
                self.task = "ollama"
                self.loaded_model_name = self.ollama_model
                return text
            except Exception as exc:
                logger.warning("Ollama generation failed: %s", exc)
                print(f"[HospitalAnswerGenerator] Ollama generation failed: {exc}")
                self.fallback_used = True

        if self._ensure_flan_generator():
            try:
                self.provider_used = "FLAN-T5"
                self.task = "text2text-generation"
                self.loaded_model_name = self.fallback_model
                out = self.generator(
                    prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                )[0]["generated_text"]
                return str(out).strip()
            except Exception as exc:
                logger.warning("FLAN fallback failed: %s", exc)
                print(f"[HospitalAnswerGenerator] FLAN fallback failed: {exc}")
                self.secondary_fallback_used = True

        try:
            self.provider_used = "Groq"
            self.task = "groq"
            self.loaded_model_name = self.groq_service.model
            self.secondary_fallback_used = True
            return self.groq_service.generate(prompt)
        except Exception as exc:
            logger.warning("Groq fallback failed: %s", exc)
            print(f"[HospitalAnswerGenerator] Groq fallback failed: {exc}")
            return "No AI provider returned a response."

    def generate(self, question: str, context: str) -> str:
        prompt = f"""
You are a hospital medicine assistant.
Use only the provided context.
If context is not sufficient, say what is missing.

Context:
{context}

Question:
{question}

Answer clearly with:
1) direct answer
2) dosage guidance from context (if present)
3) alternative medicine (if present)
"""
        return self.generate_prompt(prompt).strip()
