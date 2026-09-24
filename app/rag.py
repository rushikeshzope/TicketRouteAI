"""ChromaDB vector store setup, local embeddings, retrieval, and direct match."""

import json
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.router import RAG_SIMILARITY_THRESHOLD

PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
COLLECTION_NAME = "ticket_kb"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Global references for cached instances
_embedding_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.ClientAPI] = None
_collection: Optional[chromadb.Collection] = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load and cache the SentenceTransformer model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_chroma_collection() -> chromadb.Collection:
    """Get or initialize the persistent ChromaDB collection configured for cosine space."""
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(
            path=PERSIST_DIRECTORY,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def seed_chroma_if_empty(seed_file_path: str = "data/seed_examples.jsonl") -> int:
    """Populate ChromaDB from seed JSONL file if the collection is empty."""
    collection = get_chroma_collection()
    count = collection.count()
    if count > 0:
        return count

    if not os.path.exists(seed_file_path):
        return 0

    model = get_embedding_model()
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    with open(seed_file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = item.get("text", "").strip()
            team = item.get("team", "").strip()
            if text and team:
                documents.append(text)
                metadatas.append({"team": team, "source": "seed"})
                ids.append(f"seed_{idx}_{uuid.uuid4().hex[:8]}")

    if documents:
        embeddings = model.encode(documents, convert_to_numpy=True).tolist()
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    return collection.count()


def query_similar_tickets(
    cleaned_text: str, k: int = 3
) -> List[Dict[str, Any]]:
    """Query Chroma for top k most similar tickets and compute cosine similarity."""
    collection = get_chroma_collection()
    total_docs = collection.count()
    if total_docs == 0 or not cleaned_text:
        return []

    n_results = min(k, total_docs)
    model = get_embedding_model()
    query_embedding = model.encode([cleaned_text], convert_to_numpy=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    retrieved: List[Dict[str, Any]] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        # Cosine distance in Chroma is 1 - cosine_similarity
        # Therefore, cosine_similarity = 1 - distance
        cosine_sim = max(0.0, min(1.0, 1.0 - float(dist)))
        retrieved.append(
            {
                "text": doc,
                "team": meta.get("team", "GENERAL_TRIAGE"),
                "similarity": cosine_sim,
                "source": meta.get("source", "unknown"),
            }
        )

    return retrieved


def check_direct_rag_match(
    retrieved_examples: List[Dict[str, Any]],
    threshold: float = RAG_SIMILARITY_THRESHOLD,
) -> Optional[Tuple[str, float]]:
    """Check if the top match exceeds the direct match similarity threshold (0.85).

    Returns (team, similarity_score) if matched, else None.
    """
    if not retrieved_examples:
        return None

    top_match = retrieved_examples[0]
    similarity = top_match.get("similarity", 0.0)

    if similarity > threshold:
        return top_match.get("team", "GENERAL_TRIAGE"), similarity

    return None


def add_confirmed_ticket_to_vector_store(
    ticket_id: int, text: str, team: str
) -> None:
    """Add confirmed ticket into Chroma for future direct similarity matching."""
    if not text or not team:
        return

    collection = get_chroma_collection()
    model = get_embedding_model()
    embedding = model.encode([text], convert_to_numpy=True).tolist()

    collection.add(
        ids=[f"confirmed_ticket_{ticket_id}_{uuid.uuid4().hex[:8]}"],
        documents=[text],
        metadatas=[{"team": team, "source": f"ticket_{ticket_id}"}],
        embeddings=embedding,
    )
