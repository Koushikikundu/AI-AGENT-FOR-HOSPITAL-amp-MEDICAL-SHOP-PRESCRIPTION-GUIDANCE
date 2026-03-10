from __future__ import annotations

from dataclasses import dataclass

import chromadb
from sentence_transformers import SentenceTransformer

from backend.config import CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME, TOP_K, VECTOR_STORE_DIR


@dataclass
class RetrievedDoc:
    page_content: str
    metadata: dict
    distance: float | None = None


class ChromaMedicineRetriever:
    def __init__(
        self,
        persist_directory: str | None = None,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ) -> None:
        self.embedding_model = SentenceTransformer(embedding_model_name)
        client = chromadb.PersistentClient(path=persist_directory or str(VECTOR_STORE_DIR))
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def search(self, query: str, k: int = TOP_K) -> list[RetrievedDoc]:
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [
            RetrievedDoc(
                page_content=doc or "",
                metadata=meta or {},
                distance=dist if dist is not None else None,
            )
            for doc, meta, dist in zip(docs, metas, distances)
        ]
