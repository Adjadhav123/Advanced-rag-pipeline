from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings 
import logging

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def load_embedding_model() -> HuggingFaceEmbeddings:
    """Load embedding model."""

    logger.info(f"Loading embedding model from {settings.embedding_model_name}")

    embedding_model = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"cache_dir": settings.cache_dir},
        encode_kwargs={"normalize_embeddings": True},
    )

    return embedding_model

@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Get vector store from the directory."""

    Path(settings.chroma_dir).mkdir(exist_ok=True, parents=True)

    return Chroma(
        collection_name="production-rag",
        persist_directory=settings.chroma_dir,
        embedding_function=load_embedding_model(),
        collection_metadata={"hnsw:space":"cosine"}
    )



