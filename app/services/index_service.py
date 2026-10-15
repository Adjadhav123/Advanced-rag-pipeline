import json 
import logging
from pathlib import Path 

from app.config import settings
from app.logger import configure_logging

from app.services.document_loader import load_documents, split_documents
from app.services.vector_db import get_vector_store

logger = configure_logging()

def write_bm25_snapshot() -> int:
    """Export indexed Chroma documents for BM25 retrieval."""

    store = get_vector_store()
    result = store.get(
        include=["documents", "metadatas"]
    )

    docs = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    ids = result.get("ids") or []

    records = []

    for doc_id, texts, metadata in zip(ids, docs, metadatas):
        if not texts:
            continue 

        records.append({
            "id": doc_id,
            "text": texts,
            "metadata": metadata,
        })

    snapshot_path = Path(settings.bm25_snapshot_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    logger.info(f"Wrote {len(records)} records to BM25 snapshot: {snapshot_path}")

    return len(records)


def index_documents(directory_path: str | Path | None = None) -> int:
    """Load, split, and index documents into Chroma and BM25 snapshot."""
    target_dir = Path(directory_path) if directory_path else Path(settings.docs_dir)

    docs = load_documents(target_dir)
    chunks = split_documents(docs)

    if not chunks:
        logger.warning(f"No chunks found to index from {target_dir}.")
        return 0

    store = get_vector_store()
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    store.add_documents(chunks, ids=ids)
    logger.info(f"Indexed {len(chunks)} chunks into Chroma.")

    count = write_bm25_snapshot()
    logger.info(f"Created BM25 snapshot with {count} records.")

    return len(chunks)

