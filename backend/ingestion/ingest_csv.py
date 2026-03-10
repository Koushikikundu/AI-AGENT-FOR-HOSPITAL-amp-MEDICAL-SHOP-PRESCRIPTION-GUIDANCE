from __future__ import annotations

import argparse
import math
import time

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from backend.config import CHROMA_COLLECTION_NAME, CSV_PATH, EMBEDDING_MODEL_NAME, VECTOR_STORE_DIR
from backend.utils.csv_loader import MedicineCSVRepository


def row_to_document(row) -> tuple[str, dict]:
    # No character chunking: one structured row -> one document.
    text = (
    f"Medicine: {row['medicine_name']} {row['strength']}.\n"
    f"Use case: {row['use_case']}.\n"
    f"Dosage: {row['dosage']}.\n"
    f"Alternative medicine: {row['alternative']}.\n"
    f"Manufacturer: {row['manufacturer']}."
    )
    metadata = {
        "medicine_id": str(row["medicine_id"]),
        "medicine_name": str(row["medicine_name"]),
        "strength": str(row["strength"]),
        "use_case": str(row["use_case"]),
        "dosage": str(row["dosage"]),
        "alternative": str(row["alternative"]),
        "manufacturer": str(row["manufacturer"]),
    }
    return text, metadata


def ingest(clear_collection: bool = False) -> None:
    started_at = time.time()
    print("Loading CSV data...")
    repo = MedicineCSVRepository(str(CSV_PATH))
    df = repo.get_dataframe()
    print(f"Loaded {len(df)} rows from {CSV_PATH}.")

    rows = [row_to_document(row) for _, row in df.iterrows()]
    documents = [r[0] for r in rows]
    metadatas = [r[1] for r in rows]
    ids = [str(mid) for mid in df["medicine_id"].tolist()]

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))

    if clear_collection:
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch upserts keep memory stable for larger datasets.
    batch_size = 128
    total_batches = math.ceil(len(documents) / batch_size)
    print(f"Starting ingestion in {total_batches} batches...")
    for batch_no, start in enumerate(tqdm(range(0, len(documents), batch_size), desc="Ingesting"), start=1):
        end = start + batch_size
        batch_docs = documents[start:end]
        batch_meta = metadatas[start:end]
        batch_ids = ids[start:end]
        batch_embeddings = embedding_model.encode(
            batch_docs,
            batch_size=min(64, len(batch_docs)),
            show_progress_bar=False,
        ).tolist()
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=batch_embeddings,
        )
        if batch_no % 20 == 0 or batch_no == total_batches:
            print(f"Processed batch {batch_no}/{total_batches} ({min(end, len(documents))}/{len(documents)} rows)")

    elapsed = time.time() - started_at
    print(
        f"Ingested {len(documents)} rows into Chroma collection '{CHROMA_COLLECTION_NAME}' "
        f"in {elapsed:.1f} seconds."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest medicine CSV into ChromaDB.")
    parser.add_argument("--clear", action="store_true", help="Delete existing collection before ingesting.")
    args = parser.parse_args()
    ingest(clear_collection=args.clear)
