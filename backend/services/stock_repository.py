from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backend.config import (
    DATABASE_URL,
    MONGO_DB_NAME,
    MONGO_STOCK_COLLECTION,
    MONGO_URI,
    POSTGRES_STOCK_TABLE,
    STOCK_BACKEND,
)


@dataclass
class StockRecord:
    medicine_id: str
    stock: int
    source: str


class BaseStockRepository:
    source_name: str = "unknown"

    def get_stock(self, medicine_id: str) -> Optional[StockRecord]:
        raise NotImplementedError

    def set_stock(self, medicine_id: str, value: int) -> StockRecord:
        raise NotImplementedError

    def adjust_stock(self, medicine_id: str, delta: int) -> StockRecord:
        raise NotImplementedError

    def count_records(self) -> int:
        raise NotImplementedError

    def bulk_seed_if_empty(self, rows: list[dict]) -> int:
        raise NotImplementedError


class MongoStockRepository(BaseStockRepository):
    def __init__(self, uri: str, db_name: str, collection_name: str) -> None:
        from pymongo import MongoClient
        from pymongo.server_api import ServerApi

        self.source_name = "mongodb"
        self._client = MongoClient(uri, serverSelectionTimeoutMS=6000, server_api=ServerApi("1"))
        self._client.admin.command("ping")
        self._collection = self._client[db_name][collection_name]
        self._collection.create_index("medicine_id", unique=True)

    def get_stock(self, medicine_id: str) -> Optional[StockRecord]:
        doc = self._collection.find_one({"medicine_id": medicine_id}, {"_id": 0, "stock": 1})
        if not doc:
            return None
        return StockRecord(medicine_id=medicine_id, stock=int(doc.get("stock", 0)), source="mongodb")

    def set_stock(self, medicine_id: str, value: int) -> StockRecord:
        value = max(0, int(value))
        self._collection.update_one(
            {"medicine_id": medicine_id},
            {"$set": {"medicine_id": medicine_id, "stock": value}},
            upsert=True,
        )
        return StockRecord(medicine_id=medicine_id, stock=value, source="mongodb")

    def adjust_stock(self, medicine_id: str, delta: int) -> StockRecord:
        current = self.get_stock(medicine_id)
        base = current.stock if current else 0
        return self.set_stock(medicine_id, max(0, base + int(delta)))

    def count_records(self) -> int:
        return int(self._collection.estimated_document_count())

    def bulk_seed_if_empty(self, rows: list[dict]) -> int:
        if self.count_records() > 0:
            return 0
        from pymongo import UpdateOne

        ops = [
            UpdateOne(
                {"medicine_id": str(row["medicine_id"])},
                {
                    "$setOnInsert": {
                        "medicine_id": str(row["medicine_id"]),
                        "stock": int(row["stock"]),
                    }
                },
                upsert=True,
            )
            for row in rows
        ]
        if not ops:
            return 0
        self._collection.bulk_write(ops, ordered=False)
        return len(ops)


class PostgresStockRepository(BaseStockRepository):
    def __init__(self, database_url: str, table_name: str) -> None:
        import psycopg2

        self.source_name = "postgres"
        self._psycopg2 = psycopg2
        self._database_url = database_url
        self._table_name = table_name
        self._ensure_table()

    def _connect(self):
        return self._psycopg2.connect(self._database_url)

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table_name} (
                        medicine_id TEXT PRIMARY KEY,
                        stock INTEGER NOT NULL CHECK (stock >= 0)
                    )
                    """
                )

    def get_stock(self, medicine_id: str) -> Optional[StockRecord]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT stock FROM {self._table_name} WHERE medicine_id = %s", (medicine_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return StockRecord(medicine_id=medicine_id, stock=int(row[0]), source="postgres")

    def set_stock(self, medicine_id: str, value: int) -> StockRecord:
        value = max(0, int(value))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self._table_name} (medicine_id, stock)
                    VALUES (%s, %s)
                    ON CONFLICT (medicine_id) DO UPDATE SET stock = EXCLUDED.stock
                    """,
                    (medicine_id, value),
                )
        return StockRecord(medicine_id=medicine_id, stock=value, source="postgres")

    def adjust_stock(self, medicine_id: str, delta: int) -> StockRecord:
        current = self.get_stock(medicine_id)
        base = current.stock if current else 0
        return self.set_stock(medicine_id, max(0, base + int(delta)))

    def count_records(self) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self._table_name}")
                return int(cur.fetchone()[0])

    def bulk_seed_if_empty(self, rows: list[dict]) -> int:
        if self.count_records() > 0:
            return 0
        inserted = 0
        with self._connect() as conn:
            with conn.cursor() as cur:
                for row in rows:
                    cur.execute(
                        f"""
                        INSERT INTO {self._table_name} (medicine_id, stock)
                        VALUES (%s, %s)
                        ON CONFLICT (medicine_id) DO NOTHING
                        """,
                        (str(row["medicine_id"]), int(row["stock"])),
                    )
                    inserted += 1
        return inserted


class CsvFallbackStockRepository(BaseStockRepository):
    """Fallback repository when DB is not configured. Keeps mutable in-memory stock values."""

    def __init__(self) -> None:
        self.source_name = "csv_fallback"
        self._store: dict[str, int] = {}

    def get_stock(self, medicine_id: str) -> Optional[StockRecord]:
        if medicine_id not in self._store:
            return None
        return StockRecord(medicine_id=medicine_id, stock=int(self._store[medicine_id]), source="csv_fallback")

    def set_stock(self, medicine_id: str, value: int) -> StockRecord:
        self._store[medicine_id] = max(0, int(value))
        return StockRecord(medicine_id=medicine_id, stock=self._store[medicine_id], source="csv_fallback")

    def adjust_stock(self, medicine_id: str, delta: int) -> StockRecord:
        base = self._store.get(medicine_id, 0)
        self._store[medicine_id] = max(0, base + int(delta))
        return StockRecord(medicine_id=medicine_id, stock=self._store[medicine_id], source="csv_fallback")

    def count_records(self) -> int:
        return len(self._store)

    def bulk_seed_if_empty(self, rows: list[dict]) -> int:
        if self._store:
            return 0
        for row in rows:
            self._store[str(row["medicine_id"])] = int(row["stock"])
        return len(rows)


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

    if backend == "postgres":
        if not DATABASE_URL:
            return CsvFallbackStockRepository(), {
                "backend": "postgres",
                "connected": False,
                "source": "csv_fallback",
                "error": "DATABASE_URL not set",
            }
        try:
            repo = PostgresStockRepository(DATABASE_URL, POSTGRES_STOCK_TABLE)
            return repo, {"backend": "postgres", "connected": True, "source": repo.source_name}
        except Exception as exc:
            return CsvFallbackStockRepository(), {
                "backend": "postgres",
                "connected": False,
                "source": "csv_fallback",
                "error": str(exc),
            }

    repo = CsvFallbackStockRepository()
    return repo, {"backend": "csv", "connected": True, "source": repo.source_name}


def build_stock_repository() -> BaseStockRepository:
    repo, _ = build_stock_repository_with_status()
    return repo
