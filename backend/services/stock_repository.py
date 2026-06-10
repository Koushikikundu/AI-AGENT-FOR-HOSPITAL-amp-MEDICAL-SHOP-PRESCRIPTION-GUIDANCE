from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import re

from rapidfuzz import fuzz, process

from backend.config import (
    MONGO_DB_NAME,
    MONGO_STOCK_COLLECTION,
    MONGO_URI,
    STOCK_BACKEND,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).lower()


@dataclass
class StockRecord:
    medicine_name: str
    strength: str
    stock: int
    source: str
    medicine_id: str = ""
    created_at: str = ""
    updated_at: str = ""


class BaseStockRepository:
    source_name: str = "unknown"

    def get_stock(self, medicine_name: str, strength: str) -> Optional[StockRecord]:
        raise NotImplementedError

    def add_stock(self, medicine_name: str, strength: str, quantity: int, medicine_id: str = "") -> StockRecord:
        raise NotImplementedError

    def remove_stock(self, medicine_name: str, strength: str, quantity: int) -> StockRecord:
        raise NotImplementedError

    def count_records(self) -> int:
        raise NotImplementedError

    def sync_catalog(self, rows: list[dict]) -> int:
        raise NotImplementedError

    def list_records(self) -> list[StockRecord]:
        raise NotImplementedError

    def fuzzy_match_record(self, medicine_name: str, strength: str = "") -> Optional[StockRecord]:
        records = self.list_records()
        if not records:
            return None

        if strength:
            narrowed = [
                record
                for record in records
                if _normalize(record.strength) == _normalize(strength)
                or re.search(r"\d+(?:\.\d+)?", record.strength or "")
                == re.search(r"\d+(?:\.\d+)?", strength or "")
            ]
            if narrowed:
                records = narrowed

        choices = [f"{r.medicine_name} {r.strength}".strip() for r in records]
        query = f"{medicine_name} {strength}".strip()
        best = process.extractOne(query, choices, scorer=fuzz.token_set_ratio, score_cutoff=70)
        if not best:
            best = process.extractOne(medicine_name, [r.medicine_name for r in records], scorer=fuzz.token_set_ratio, score_cutoff=70)
            if not best:
                return None
            _, _, idx = best
            return records[idx]
        _, _, idx = best
        return records[idx]


class MongoStockRepository(BaseStockRepository):
    def __init__(self, uri: str, db_name: str, collection_name: str) -> None:
        from pymongo import MongoClient
        from pymongo.server_api import ServerApi

        self.source_name = "mongodb"
        self._client = MongoClient(uri, serverSelectionTimeoutMS=6000, server_api=ServerApi("1"))
        self._client.admin.command("ping")
        self._collection = self._client[db_name][collection_name]
        self._collection.update_many({"medicine_id": {"$in": ["", None]}}, {"$unset": {"medicine_id": ""}})
        existing_indexes = self._collection.index_information()
        if "medicine_id_1" in existing_indexes:
            self._collection.drop_index("medicine_id_1")
        self._collection.create_index(
            [("medicine_id", 1)],
            unique=True,
            partialFilterExpression={
                "medicine_id": {"$exists": True, "$type": "string"},
            },
        )
        self._collection.create_index(
            [("medicine_name", 1), ("strength", 1)],
            unique=True,
            partialFilterExpression={
                "medicine_name": {"$type": "string"},
                "strength": {"$type": "string"},
            },
        )

    @staticmethod
    def _to_record(doc: dict) -> StockRecord:
        return StockRecord(
            medicine_name=str(doc.get("medicine_name", "")),
            strength=str(doc.get("strength", "")),
            stock=int(doc.get("stock", 0)),
            source="mongodb",
            medicine_id=str(doc.get("medicine_id", "")),
            created_at=str(doc.get("createdAt", "")),
            updated_at=str(doc.get("updatedAt", "")),
        )

    def _find_doc(self, medicine_name: str, strength: str) -> Optional[dict]:
        return self._collection.find_one(
            {
                "medicine_name": {"$regex": f"^{re.escape(medicine_name.strip())}$", "$options": "i"},
                "strength": {"$regex": f"^{re.escape(strength.strip())}$", "$options": "i"},
            },
            sort=[("updatedAt", -1), ("createdAt", -1)],
        )

    def get_stock(self, medicine_name: str, strength: str) -> Optional[StockRecord]:
        doc = self._find_doc(medicine_name, strength)
        return self._to_record(doc) if doc else None

    def add_stock(self, medicine_name: str, strength: str, quantity: int, medicine_id: str = "") -> StockRecord:
        quantity = max(0, int(quantity))
        now = _utc_now()
        doc = self._find_doc(medicine_name, strength)
        if doc:
            new_stock = int(doc.get("stock", 0)) + quantity
            self._collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"stock": new_stock, "updatedAt": now}},
            )
            doc["stock"] = new_stock
            doc["updatedAt"] = now
            return self._to_record(doc)

        new_doc = {
            "medicine_name": medicine_name.strip(),
            "strength": strength.strip(),
            "stock": quantity,
            "createdAt": now,
            "updatedAt": now,
        }
        if medicine_id:
            new_doc["medicine_id"] = medicine_id
        self._collection.insert_one(new_doc)
        return self._to_record(new_doc)

    def remove_stock(self, medicine_name: str, strength: str, quantity: int) -> StockRecord:
        quantity = max(0, int(quantity))
        doc = self._find_doc(medicine_name, strength)
        if not doc:
            raise ValueError("Medicine not found in stock database.")
        current_stock = int(doc.get("stock", 0))
        if quantity > current_stock:
            raise ValueError("Insufficient stock")
        new_stock = current_stock - quantity
        now = _utc_now()
        self._collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"stock": new_stock, "updatedAt": now}},
        )
        doc["stock"] = new_stock
        doc["updatedAt"] = now
        return self._to_record(doc)

    def count_records(self) -> int:
        return int(self._collection.estimated_document_count())

    def sync_catalog(self, rows: list[dict]) -> int:
        from pymongo import UpdateOne

        ops = []
        now = _utc_now()
        for row in rows:
            medicine_id = str(row.get("medicine_id", ""))
            filter_doc = {"medicine_name": row["medicine_name"], "strength": row["strength"]}
            if medicine_id:
                filter_doc = {"$or": [{"medicine_id": medicine_id}, filter_doc]}
            set_doc = {
                "medicine_name": row["medicine_name"],
                "strength": row["strength"],
                "updatedAt": now,
            }
            if medicine_id:
                set_doc["medicine_id"] = medicine_id
            ops.append(
                UpdateOne(
                    filter_doc,
                    {
                        "$set": set_doc,
                        "$setOnInsert": {
                            "stock": int(row["stock"]),
                            "createdAt": now,
                        },
                    },
                    upsert=True,
                )
            )
        if ops:
            self._collection.bulk_write(ops, ordered=False)
        return len(ops)

    def list_records(self) -> list[StockRecord]:
        return [self._to_record(doc) for doc in self._collection.find({}, {"_id": 0})]


class CsvFallbackStockRepository(BaseStockRepository):
    def __init__(self) -> None:
        self.source_name = "csv_fallback"
        self._store: dict[tuple[str, str], StockRecord] = {}

    def get_stock(self, medicine_name: str, strength: str) -> Optional[StockRecord]:
        return self._store.get((_normalize(medicine_name), _normalize(strength)))

    def add_stock(self, medicine_name: str, strength: str, quantity: int, medicine_id: str = "") -> StockRecord:
        key = (_normalize(medicine_name), _normalize(strength))
        current = self._store.get(key)
        if current:
            current.stock += max(0, int(quantity))
            current.updated_at = str(_utc_now())
            return current
        record = StockRecord(
            medicine_name=medicine_name.strip(),
            strength=strength.strip(),
            stock=max(0, int(quantity)),
            source="csv_fallback",
            medicine_id=medicine_id,
            created_at=str(_utc_now()),
            updated_at=str(_utc_now()),
        )
        self._store[key] = record
        return record

    def remove_stock(self, medicine_name: str, strength: str, quantity: int) -> StockRecord:
        current = self.get_stock(medicine_name, strength)
        if not current:
            raise ValueError("Medicine not found in stock database.")
        if int(quantity) > current.stock:
            raise ValueError("Insufficient stock")
        current.stock -= int(quantity)
        current.updated_at = str(_utc_now())
        return current

    def count_records(self) -> int:
        return len(self._store)

    def sync_catalog(self, rows: list[dict]) -> int:
        for row in rows:
            key = (_normalize(row["medicine_name"]), _normalize(row["strength"]))
            if key not in self._store:
                self._store[key] = StockRecord(
                    medicine_name=row["medicine_name"],
                    strength=row["strength"],
                    stock=int(row["stock"]),
                    source="csv_fallback",
                    medicine_id=str(row.get("medicine_id", "")),
                    created_at=str(_utc_now()),
                    updated_at=str(_utc_now()),
                )
        return len(rows)

    def list_records(self) -> list[StockRecord]:
        return list(self._store.values())


def build_stock_repository_with_status() -> tuple[BaseStockRepository, dict]:
    backend = STOCK_BACKEND

    if backend == "mongodb":
        if not MONGO_URI:
            return CsvFallbackStockRepository(), {
                "backend": "mongodb",
                "connected": False,
                "source": "csv_fallback",
                "error": "MONGO_URI not set",
            }
        try:
            repo = MongoStockRepository(MONGO_URI, MONGO_DB_NAME, MONGO_STOCK_COLLECTION)
            return repo, {"backend": "mongodb", "connected": True, "source": repo.source_name}
        except Exception as exc:
            return CsvFallbackStockRepository(), {
                "backend": "mongodb",
                "connected": False,
                "source": "csv_fallback",
                "error": str(exc),
            }

    repo = CsvFallbackStockRepository()
    return repo, {"backend": "csv", "connected": True, "source": repo.source_name}
