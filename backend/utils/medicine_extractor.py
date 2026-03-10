from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from transformers import pipeline

from backend.config import EXTRACTION_MODEL_NAME, STOCK_INTENT_KEYWORDS


@dataclass
class QueryExtraction:
    intent: str
    medicine_name: str
    strength: str
    raw_model_output: str


class MedicineExtractor:
    """
    Extracts intent + medicine details from informal user queries.
    Uses a lightweight HF model and falls back to regex heuristics.
    """

    def __init__(self, model_name: str = EXTRACTION_MODEL_NAME) -> None:
        self.model_name = model_name
        self._pipe = pipeline("text2text-generation", model=model_name)

    @staticmethod
    def _is_stock_intent(query: str) -> bool:
        q = query.lower()
        return any(k in q for k in STOCK_INTENT_KEYWORDS)

    @staticmethod
    def _extract_strength_with_regex(query: str) -> str:
        match = re.search(r"\b\d+\s?(mg|ml|g|mcg)\b", query.lower())
        return match.group(0) if match else ""

    @staticmethod
    def _clean_medicine_name_candidate(query: str, strength: str) -> str:
        cleaned = query.lower()
        if strength:
            cleaned = cleaned.replace(strength.lower(), " ")

        # Remove common intent/noise words so fuzzy matching targets the medicine token(s).
        stop_phrases = [
            "do we have",
            "in stock",
            "stock of",
            "stock",
            "available",
            "availability",
            "what is",
            "used for",
            "alternative for",
            "alternative",
            "for",
            "of",
            "the",
            "a",
            "an",
            "please",
        ]
        for phrase in stop_phrases:
            cleaned = cleaned.replace(phrase, " ")
        cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _safe_json_parse(text: str) -> dict:
        # Extract first JSON object from model output.
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    def extract(self, query: str) -> QueryExtraction:
        prompt = f"""
You are an information extraction system for medicine queries.
Return only valid JSON with keys:
- intent: one of ["stock","knowledge"]
- medicine_name: string
- strength: string

Query: "{query}"
"""
        output = self._pipe(prompt, max_new_tokens=96, do_sample=False)[0]["generated_text"].strip()
        parsed = self._safe_json_parse(output)

        intent = parsed.get("intent", "").strip().lower()
        if intent not in {"stock", "knowledge"}:
            intent = "stock" if self._is_stock_intent(query) else "knowledge"

        medicine_name = parsed.get("medicine_name", "").strip()
        strength = parsed.get("strength", "").strip()
        if not strength:
            strength = self._extract_strength_with_regex(query)

        # Guardrail fallback: if model misses name, use query minus strength tokens.
        if not medicine_name:
            medicine_name = self._clean_medicine_name_candidate(query, strength)

        return QueryExtraction(
            intent=intent,
            medicine_name=medicine_name,
            strength=strength,
            raw_model_output=output,
        )
