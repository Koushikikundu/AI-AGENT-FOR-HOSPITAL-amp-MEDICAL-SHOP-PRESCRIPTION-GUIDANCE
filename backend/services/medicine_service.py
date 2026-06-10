from __future__ import annotations

import io
import re

import pandas as pd

from backend.rag.rag_pipeline import HospitalRAGPipeline
from backend.services.stock_repository import BaseStockRepository, StockRecord, build_stock_repository_with_status
from backend.utils.csv_loader import MedicineCSVRepository
from backend.utils.medicine_extractor import MedicineExtractor


class MedicineAssistantService:
    """Route stock to DB and knowledge to Chroma/AI fallback."""

    def __init__(self) -> None:
        self.csv_repo = MedicineCSVRepository()
        self.extractor = MedicineExtractor()
        self.rag_pipeline = HospitalRAGPipeline()
        self.stock_repo, self.stock_backend_status = build_stock_repository_with_status()
        self.synced_rows = self._sync_stock_catalog()

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
                "drug interactions",
                "side effects",
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
            "side effects",
            "drug interactions",
        ]:
            cleaned = cleaned.replace(token, " ")
        cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _sync_stock_catalog(self) -> int:
        rows = self.csv_repo.get_dataframe()[["medicine_id", "medicine_name", "strength", "stock"]].to_dict(
            orient="records"
        )
        try:
            return int(self.stock_repo.sync_catalog(rows))
        except Exception:
            return 0

    def _resolve_stock_record(self, query: str, extracted_name: str, extracted_strength: str) -> StockRecord | None:
        strength = extracted_strength or self._extract_strength_from_query(query)
        candidates: list[tuple[str, str]] = []
        if extracted_name:
            candidates.append((extracted_name, strength))
        cleaned = self._clean_name_from_query(query, strength)
        if cleaned:
            candidates.append((cleaned, strength))

        # Stock must be deterministic. We only return a record on an exact
        # medicine_name + strength match in the stock database.
        for name, s in candidates:
            if not name or not s:
                continue
            record = self.stock_repo.get_stock(name, s)
            if record is not None:
                return record
        return None

    def _handle_stock_query(self, query: str, extracted_name: str, extracted_strength: str) -> dict:
        record = self._resolve_stock_record(query, extracted_name, extracted_strength)
        if record is None:
            return {
                "route": "stock",
                "lookup_source": self.stock_backend_status.get("source"),
                "answer": "Stock unavailable for the requested medicine and strength.",
                "matched_medicine": None,
                "stock_backend_status": self.stock_backend_status,
            }

        stock_status = "In stock" if record.stock > 0 else "Out of stock"
        return {
            "route": "stock",
            "lookup_source": record.source,
            "answer": (
                f"{record.medicine_name} {record.strength}: {stock_status}. "
                f"Current quantity: {record.stock} units."
            ),
            "matched_medicine": {
                "medicine_id": record.medicine_id,
                "medicine_name": record.medicine_name,
                "strength": record.strength,
                "stock": record.stock,
            },
            "stock_backend_status": self.stock_backend_status,
            "synced_rows_on_startup": self.synced_rows,
        }

    def add_stock(self, medicine_name: str, strength: str, quantity: int) -> dict:
        if quantity <= 0:
            return {"ok": False, "message": "Quantity must be greater than zero."}
        record = self.stock_repo.add_stock(medicine_name.strip(), strength.strip(), int(quantity))
        return {
            "ok": True,
            "message": f"Updated stock for {record.medicine_name} {record.strength}: {record.stock}",
            "medicine_name": record.medicine_name,
            "strength": record.strength,
            "stock": record.stock,
            "stock_source": record.source,
            "stock_backend_status": self.stock_backend_status,
        }

    def remove_stock(self, medicine_name: str, strength: str, quantity: int) -> dict:
        if quantity <= 0:
            return {"ok": False, "message": "Quantity must be greater than zero."}
        try:
            record = self.stock_repo.remove_stock(medicine_name.strip(), strength.strip(), int(quantity))
            return {
                "ok": True,
                "message": f"Updated stock for {record.medicine_name} {record.strength}: {record.stock}",
                "medicine_name": record.medicine_name,
                "strength": record.strength,
                "stock": record.stock,
                "stock_source": record.source,
                "stock_backend_status": self.stock_backend_status,
            }
        except Exception as exc:
            return {"ok": False, "message": str(exc), "stock_backend_status": self.stock_backend_status}

    def update_stock(self, medicine_name: str, strength: str, delta: int) -> dict:
        if delta >= 0:
            return self.add_stock(medicine_name, strength, delta)
        return self.remove_stock(medicine_name, strength, abs(delta))

    def upload_stock_csv(self, file_bytes: bytes) -> dict:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as exc:
            return {"ok": False, "message": f"Invalid CSV file: {exc}"}

        df.columns = [str(col).strip() for col in df.columns]
        required = {"medicine_name", "strength", "stock"}
        optional = {"manufacturer_company_name"}
        missing = required - set(df.columns)
        if missing:
            return {"ok": False, "message": f"Missing required columns: {sorted(missing)}"}

        df = df.dropna(how="all")
        if df.empty:
            return {"ok": False, "message": "CSV contains no usable rows."}

        processed = 0
        new_added = 0
        updated = 0
        seen_keys: set[tuple[str, str]] = set()

        for _, row in df.iterrows():
            medicine_name = str(row.get("medicine_name", "")).strip()
            strength = str(row.get("strength", "")).strip()
            stock_raw = row.get("stock", "")
            if not medicine_name or not strength or str(stock_raw).strip() == "":
                continue

            try:
                quantity = int(float(stock_raw))
            except Exception:
                return {"ok": False, "message": f"Invalid stock value for {medicine_name} {strength}: {stock_raw}"}
            if quantity < 0:
                return {"ok": False, "message": f"Negative quantity not allowed for {medicine_name} {strength}"}

            key = (medicine_name.lower(), strength.lower())
            if key in seen_keys:
                continue
            seen_keys.add(key)

            existing = self.stock_repo.get_stock(medicine_name, strength)
            record = self.stock_repo.add_stock(medicine_name, strength, quantity)
            if existing is None:
                new_added += 1
            else:
                updated += 1
            processed += 1

        return {
            "ok": True,
            "message": "CSV upload completed.",
            "new_medicines_added": new_added,
            "existing_medicines_updated": updated,
            "total_processed": processed,
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
                result = self.rag_pipeline.answer_knowledge_query(
                    query,
                    medicine_name=parsed_name,
                    strength=parsed_strength,
                    strict_match=True,
                )
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
