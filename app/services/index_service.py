from attr.filters import include
from langchain_community.document_loaders import college_confidential
import json 
from pathlib import Path 

from app.config import settings
from app.logger import configure_logging

from app.services.document_loader import load_documents, split_documents
from app.services.vector_db import get_vector_store

def write_bm25_snapshot() -> int:
    """Export indexed Chroma documents for BM25 retrieval."""

    store = get_vector_store()
    collection = store.get(collection)

    result = collection.get(
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
    
    
    

