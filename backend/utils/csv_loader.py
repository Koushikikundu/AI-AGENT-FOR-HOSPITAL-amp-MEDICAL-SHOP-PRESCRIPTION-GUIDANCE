from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional
import re

import pandas as pd
from rapidfuzz import fuzz, process

from backend.config import CSV_PATH, FUZZY_SCORE_CUTOFF


@dataclass
class MatchedMedicine:
    medicine_id: str
    medicine_name: str
    strength: str
    use_case: str
    alternative: str
    stock: int
    dosage: str
    manufacturer: str
    fuzzy_score: float


class MedicineCSVRepository:
    """Loads medicine data once and provides structured lookup helpers."""

    def __init__(self, csv_path: str | None = None) -> None:
        self.csv_path = csv_path or str(CSV_PATH)
        self._df = self._load_csv(self.csv_path)

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_csv(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        # Support either `use_case` or `usage` in source CSV.
        if "use_case" not in df.columns and "usage" in df.columns:
            df = df.rename(columns={"usage": "use_case"})

        required_columns = {
            "medicine_id",
            "medicine_name",
            "strength",
            "use_case",
            "alternative",
            "stock",
            "dosage",
            "manufacturer",
        }
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        # Normalize text columns for robust matching
        text_cols = ["medicine_name", "strength", "use_case", "alternative", "dosage", "manufacturer"]
        for col in text_cols:
            df[col] = df[col].fillna("").astype(str)
        df["stock"] = pd.to_numeric(df["stock"], errors="coerce").fillna(0).astype(int)
        return df

    def get_dataframe(self) -> pd.DataFrame:
        return self._df

    def _candidate_name_strength(self) -> pd.Series:
        return (self._df["medicine_name"] + " " + self._df["strength"]).str.strip()

    @staticmethod
    def _extract_strength_number(strength: str) -> str:
        match = re.search(r"\d+(?:\.\d+)?", strength.lower())
        return match.group(0) if match else ""

    def fuzzy_match(
        self,
        medicine_name: str,
        strength: Optional[str] = None,
        strict_strength: bool = False,
        allow_numeric_strength_fallback: bool = False,
    ) -> Optional[MatchedMedicine]:
        if not medicine_name:
            return None

        query = medicine_name.strip()
        name_strength_query = query if not strength else f"{query} {strength}".strip()
        search_df = self._df

        # If strength is provided, optionally enforce exact normalized strength.
        if strength:
            s_norm = strength.strip().lower().replace(" ", "")
            strength_series = search_df["strength"].str.lower().str.replace(" ", "", regex=False)
            narrowed = search_df[strength_series == s_norm]
            if narrowed.empty and strict_strength:
                if allow_numeric_strength_fallback:
                    requested_num = self._extract_strength_number(strength)
                    if requested_num:
                        numeric_series = search_df["strength"].apply(self._extract_strength_number)
                        numeric_narrowed = search_df[numeric_series == requested_num]
                        if not numeric_narrowed.empty:
                            search_df = numeric_narrowed
                        else:
                            return None
                    else:
                        return None
                else:
                    return None
            if not narrowed.empty:
                search_df = narrowed

        choices = (search_df["medicine_name"] + " " + search_df["strength"]).str.strip().tolist()
        best = process.extractOne(
            name_strength_query,
            choices,
            scorer=fuzz.token_set_ratio,
            score_cutoff=FUZZY_SCORE_CUTOFF,
        )

        # Fallback to medicine_name-only match when strength is missing/noisy.
        if best is None and strict_strength and strength:
            return None
        if best is None:
            best = process.extractOne(
                query,
                search_df["medicine_name"].tolist(),
                scorer=fuzz.token_set_ratio,
                score_cutoff=FUZZY_SCORE_CUTOFF,
            )
            if best is None:
                return None
            _, score, idx = best
            row = search_df.iloc[idx]
        else:
            _, score, idx = best
            row = search_df.iloc[idx]
        return MatchedMedicine(
            medicine_id=str(row["medicine_id"]),
            medicine_name=row["medicine_name"],
            strength=row["strength"],
            use_case=row["use_case"],
            alternative=row["alternative"],
            stock=int(row["stock"]),
            dosage=row["dosage"],
            manufacturer=row["manufacturer"],
            fuzzy_score=float(score),
        )
