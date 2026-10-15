import os 
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value.strip()) if value and value.strip() else default 

def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value.strip()) if value and value.strip() else default


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "").strip()
    groq_model: str = os.getenv("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile"

    docs_dir: str = os.getenv("DOCS_DIR", "").strip() or "docs"
    chroma_dir: str = os.getenv("CHROMA_DIR", "").strip() or "storage/chroma"
    bm25_snapshot_path: str = os.getenv("BM25_SNAPSHOT_PATH", "").strip() or "storage/bm25.json"

    logging_dir: str = os.getenv("LOGGING_DIR", "").strip() or "logs"
    cache_dir: str | None = os.getenv("CACHE_DIR", "").strip() or None

    embedding_model: str = (
        os.getenv("EMBEDDING_MODEL", "").strip()
        or os.getenv("EMBEDDING_MODEL_NAME", "").strip()
        or "sentence-transformers/all-MiniLM-L6-v2"
    )
    reranker_model: str = (
        os.getenv("RERANKER_MODEL", "").strip()
        or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    api_key: str = os.getenv("API_KEY", "").strip()

    chunk_size: int = env_int("CHUNK_SIZE", 900)
    chunk_overlap: int = env_int("CHUNK_OVERLAP", 150)  

    bm25_top_k: int = env_int("BM25_TOP_K", 20)
    rerank_top_k: int = env_int("RERANK_TOP_K", 40)    
    max_content_chars: int = env_int("MAX_CONTENT_CHARS", 1024)

    max_history_turns: int = env_int("MAX_HISTORY_TURNS", 4)

    @property
    def embedding_model_name(self) -> str:
        return self.embedding_model

    def validate(self) -> None:
        """Validate critical configuration settings."""
        pass


settings = Settings()
    

