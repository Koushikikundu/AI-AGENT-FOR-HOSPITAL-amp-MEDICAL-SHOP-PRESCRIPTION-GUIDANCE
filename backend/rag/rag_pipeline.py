from __future__ import annotations

import re

from rapidfuzz import fuzz

from backend.rag.llm_model import HospitalAnswerGenerator
from backend.rag.retriever import ChromaMedicineRetriever


class HospitalRAGPipeline:
    def __init__(self) -> None:
        self.retriever = ChromaMedicineRetriever()
        self.answer_model = HospitalAnswerGenerator()

    @staticmethod
    def _build_context_from_docs(docs: list) -> str:
        if not docs:
            return "No relevant medicine information found."
        return "\n\n".join(f"- {d.page_content}" for d in docs)

    @staticmethod
    def _extract_use_case_from_doc(doc) -> str:
        md = doc.metadata or {}
        if md.get("use_case"):
            return str(md.get("use_case"))
        text = doc.page_content or ""
        m = re.search(r"Use case:\s*(.+?)(?:\.|\n|$)", text, flags=re.IGNORECASE)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _norm_strength(s: str) -> str:
        return (s or "").lower().replace(" ", "")

    @staticmethod
    def _strength_num(s: str) -> str:
        m = re.search(r"\d+(?:\.\d+)?", (s or "").lower())
        return m.group(0) if m else ""

    def _select_best_doc(self, docs: list, medicine_name: str, strength: str):
        if not docs:
            return None, -1.0
        name_query = (medicine_name or "").strip().lower()
        str_query = self._norm_strength(strength)

        best = None
        best_score = -1.0
        for d in docs:
            md = d.metadata or {}
            name = str(md.get("medicine_name", "")).lower()
            doc_strength = str(md.get("strength", ""))
            score = float(fuzz.token_set_ratio(name_query, name)) if name_query else 0.0
            if str_query:
                doc_norm = self._norm_strength(doc_strength)
                if doc_norm == str_query:
                    score += 30.0
                elif self._strength_num(doc_strength) == self._strength_num(strength):
                    score += 10.0
            if score > best_score:
                best = d
                best_score = score
        return best, best_score

    def answer_structured_from_chroma(
        self,
        query: str,
        field_name: str,
        medicine_name: str,
        strength: str,
    ) -> dict | None:
        query_for_search = query
        if medicine_name:
            query_for_search = f"{query} {medicine_name} {strength}".strip()
        docs = self.retriever.search(query_for_search, k=25)
        best_doc, best_score = self._select_best_doc(docs, medicine_name, strength)
        if best_doc is None:
            return None

        # Prevent hallucinated cross-medicine answers.
        # Require enough name similarity when user provided a medicine name.
        if medicine_name and best_score < 70:
            return None

        label_map = {
            "alternative": "Alternative",
            "dosage": "Dosage",
            "manufacturer": "Manufacturer",
            "use_case": "Use case",
        }
        md = best_doc.metadata or {}
        value = str(md.get(field_name, "")).strip()
        if not value:
            return None
        return {
            "route": "structured_knowledge",
            "lookup_source": "chroma",
            "field": field_name,
            "field_value": value,
            "answer": f"{label_map.get(field_name, field_name)} for {md.get('medicine_name','')} {md.get('strength','')}: {value}",
            "matched_medicine": {
                "medicine_id": md.get("medicine_id"),
                "medicine_name": md.get("medicine_name"),
                "strength": md.get("strength"),
            },
            "match_score": best_score,
            "retrieved_docs": [
                {
                    "medicine_id": d.metadata.get("medicine_id"),
                    "medicine_name": d.metadata.get("medicine_name"),
                    "strength": d.metadata.get("strength"),
                }
                for d in docs
            ],
        }

    def answer_knowledge_query(
        self,
        query: str,
        medicine_name: str = "",
        strength: str = "",
        strict_match: bool = False,
    ) -> dict:
        docs = self.retriever.search(query)
        best_doc, best_score = self._select_best_doc(docs, medicine_name, strength)
        if strict_match and medicine_name and (best_doc is None or best_score < 70):
            return {
                "route": "knowledge",
                "lookup_source": "chroma",
                "answer": "Medicine not found in knowledge base for the requested name/strength.",
                "retrieved_docs": [],
                "not_found": True,
            }

        context = self._build_context_from_docs(docs)
        answer = self.answer_model.generate(query, context)
        return {
            "route": "knowledge",
            "lookup_source": "chroma+llm",
            "answer": answer,
            "match_score": best_score if best_doc is not None else None,
            "retrieved_docs": [
                {
                    "medicine_id": d.metadata.get("medicine_id"),
                    "medicine_name": d.metadata.get("medicine_name"),
                    "strength": d.metadata.get("strength"),
                }
                for d in docs
            ],
        }

    def answer_recommendation_query(self, query: str) -> dict:
        docs = self.retriever.search(query, k=10)
        if not docs:
            return {
                "route": "knowledge",
                "lookup_source": "chroma",
                "answer": "Medicine not found in knowledge base for this symptom/query.",
                "retrieved_docs": [],
                "not_found": True,
            }

        seen = set()
        suggestions = []
        for d in docs:
            md = d.metadata or {}
            name = str(md.get("medicine_name", "")).strip()
            strength = str(md.get("strength", "")).strip()
            use_case = self._extract_use_case_from_doc(d)
            key = (name.lower(), strength.lower())
            if not name or key in seen:
                continue
            seen.add(key)
            suggestions.append(
                {
                    "medicine_name": name,
                    "strength": strength,
                    "use_case": use_case,
                }
            )
            if len(suggestions) >= 3:
                break

        if not suggestions:
            return {
                "route": "knowledge",
                "lookup_source": "chroma",
                "answer": "Medicine not found in knowledge base for this symptom/query.",
                "retrieved_docs": [],
                "not_found": True,
            }

        lines = [
            f"{idx}. {s['medicine_name']} {s['strength']} - use case: {s['use_case'] or 'not specified'}"
            for idx, s in enumerate(suggestions, start=1)
        ]
        return {
            "route": "knowledge",
            "lookup_source": "chroma",
            "answer": "Top medicines from knowledge base:\n" + "\n".join(lines),
            "recommendations": suggestions,
            "retrieved_docs": [
                {
                    "medicine_id": d.metadata.get("medicine_id"),
                    "medicine_name": d.metadata.get("medicine_name"),
                    "strength": d.metadata.get("strength"),
                }
                for d in docs
            ],
        }
