from __future__ import annotations

import re

from backend.rag.rag_pipeline import HospitalRAGPipeline
from backend.services.stock_repository import build_stock_repository_with_status
from backend.utils.csv_loader import MedicineCSVRepository
from backend.utils.medicine_extractor import MedicineExtractor


class MedicineAssistantService:
    """Route stock to DB and all other knowledge to Chroma."""

    def __init__(self) -> None:
        self.csv_repo = MedicineCSVRepository()
        self.extractor = MedicineExtractor()
        self.rag_pipeline = HospitalRAGPipeline()
        self.stock_repo, self.stock_backend_status = build_stock_repository_with_status()
        self.seeded_rows = self._seed_stock_if_empty()

    @staticmethod
    def _contains_phrase(query: str, phrases: list[str]) -> bool:
        q = query.lower()
        for phrase in phrases:
            pattern = r"\b" + re.escape(phrase.lower()).replace(r"\ ", r"\s+") + r"\b"
            if re.search(pattern, q):
                return True
        return False

    @classmethod
    def _detect_structured_field(cls, query: str) -> str | None:
        if cls._contains_phrase(query, ["alternative", "substitute", "replacement", "replace", "instead of"]):
            return "alternative"
        if cls._contains_phrase(query, ["dosage", "dose", "how much"]):
            return "dosage"
        if cls._contains_phrase(query, ["manufacturer", "company", "made by"]):
            return "manufacturer"
        if cls._contains_phrase(query, ["used for", "use case", "use of", "usage", "usages", "uses", "treat"]):
            return "use_case"
        return None

    @classmethod
    def _is_semantic_recommendation_query(cls, query: str) -> bool:
        return cls._contains_phrase(
            query,
            [
                "suggest",
                "recommend",
                "best medicine",
                "medicine for",
                "for allergy",
                "for fever",
                "for pain",
                "for cold",
                "for cough",
                "symptom",
            ],
        )

    @classmethod
    def _is_explicit_medicine_query(cls, query: str) -> bool:
        return cls._contains_phrase(
            query,
            [
                "about",
                "what is",
                "use of",
                "used for",
                "alternative for",
                "dosage of",
                "manufacturer of",
            ],
        )

    @staticmethod
    def _extract_strength_from_query(query: str) -> str:
        match = re.search(r"\b\d+\s?(mg|ml|g|mcg)\b", query.lower())
        return match.group(0) if match else ""

    @staticmethod
    def _clean_name_from_query(query: str, strength: str) -> str:
        cleaned = query.lower()
        if strength:
            cleaned = cleaned.replace(strength.lower(), " ")
        for token in [
            "alternative",
            "medicine",
            "for",
            "stock",
            "in stock",
            "available",
            "availability",
            "used for",
            "use of",
            "use case",
            "usage",
            "usages",
            "dosage",
            "dose",
            "manufacturer",
            "company",
            "what is",
            "do we have",
            "how much",
            "instead of",
        ]:
            cleaned = cleaned.replace(token, " ")
        cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _resolve_match(self, query: str, extracted_name: str, extracted_strength: str):
        strength = extracted_strength or self._extract_strength_from_query(query)
        candidates: list[tuple[str, str]] = []

        if extracted_name:
            candidates.extend([(extracted_name, strength), (extracted_name, "")])
        cleaned = self._clean_name_from_query(query, strength)
        if cleaned:
            candidates.extend([(cleaned, strength), (cleaned, "")])

        best = None
        for name, s in candidates:
            m = self.csv_repo.fuzzy_match(
                name,
                s,
                strict_strength=bool(s),
                allow_numeric_strength_fallback=True,
            )
            if m is None:
                continue
            if best is None or m.fuzzy_score > best.fuzzy_score:
                best = m
        return best

    def _seed_stock_if_empty(self) -> int:
        rows = self.csv_repo.get_dataframe()[["medicine_id", "stock"]].to_dict(orient="records")
        try:
            return int(self.stock_repo.bulk_seed_if_empty(rows))
        except Exception:
            return 0

    def _handle_stock_query(self, query: str, extracted_name: str, extracted_strength: str) -> dict:
        match = self._resolve_match(query, extracted_name, extracted_strength)
        if not match:
            return {
                "route": "stock",
                "lookup_source": self.stock_backend_status.get("source"),
                "answer": "I could not find an exact medicine+strength match in stock.",
                "matched_medicine": None,
                "stock_backend_status": self.stock_backend_status,
            }

        record = self.stock_repo.get_stock(match.medicine_id)
        if record is None:
            record = self.stock_repo.set_stock(match.medicine_id, int(match.stock))
        stock_status = "In stock" if record.stock > 0 else "Out of stock"
        return {
            "route": "stock",
            "lookup_source": record.source,
            "answer": (
                f"{match.medicine_name} {match.strength}: {stock_status}. "
                f"Current quantity: {record.stock} units. "
                f"Manufacturer: {match.manufacturer}."
            ),
            "matched_medicine": {
                "medicine_id": match.medicine_id,
                "medicine_name": match.medicine_name,
                "strength": match.strength,
                "stock": record.stock,
                "fuzzy_score": match.fuzzy_score,
            },
            "stock_backend_status": self.stock_backend_status,
            "seeded_rows_on_startup": self.seeded_rows,
        }

    def update_stock(self, medicine_name: str, strength: str, delta: int) -> dict:
        query = f"stock for {medicine_name} {strength}".strip()
        match = self._resolve_match(query, medicine_name, strength)
        if not match:
            return {
                "ok": False,
                "message": "Medicine/strength not found in master data.",
                "stock_backend_status": self.stock_backend_status,
            }

        current = self.stock_repo.get_stock(match.medicine_id)
        if current is None:
            current = self.stock_repo.set_stock(match.medicine_id, int(match.stock))
        updated = self.stock_repo.adjust_stock(match.medicine_id, int(delta))
        return {
            "ok": True,
            "message": f"Updated stock for {match.medicine_name} {match.strength}: {updated.stock}",
            "medicine_id": match.medicine_id,
            "medicine_name": match.medicine_name,
            "strength": match.strength,
            "stock": updated.stock,
            "stock_source": updated.source,
            "stock_backend_status": self.stock_backend_status,
        }

    def ask(self, query: str) -> dict:
        extraction = self.extractor.extract(query)
        structured_field = self._detect_structured_field(query)

        if extraction.intent == "stock":
            result = self._handle_stock_query(query, extraction.medicine_name, extraction.strength)
        elif structured_field is not None:
            parsed_strength = extraction.strength or self._extract_strength_from_query(query)
            parsed_name = self._clean_name_from_query(query, parsed_strength) or extraction.medicine_name
            result = self.rag_pipeline.answer_structured_from_chroma(
                query=query,
                field_name=structured_field,
                medicine_name=parsed_name,
                strength=parsed_strength,
            )
            if result is None:
                result = {
                    "route": "structured_knowledge",
                    "lookup_source": "chroma",
                    "answer": "Medicine not found in knowledge base for the requested name/strength.",
                    "matched_medicine": None,
                    "not_found": True,
                }
        else:
            parsed_strength = extraction.strength or self._extract_strength_from_query(query)
            parsed_name = self._clean_name_from_query(query, parsed_strength) or extraction.medicine_name
            if self._is_semantic_recommendation_query(query):
                result = self.rag_pipeline.answer_recommendation_query(query)
            else:
                strict_match = self._is_explicit_medicine_query(query) and not self._is_semantic_recommendation_query(
                    query
                )
                result = self.rag_pipeline.answer_knowledge_query(
                    query,
                    medicine_name=parsed_name,
                    strength=parsed_strength,
                    strict_match=strict_match,
                )

        result["extraction"] = {
            "intent": extraction.intent,
            "medicine_name": extraction.medicine_name,
            "strength": extraction.strength,
        }
        result["stock_backend_status"] = self.stock_backend_status
        return result
