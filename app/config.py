import os 
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default 

def env_float(name: str, default: float) -> int:
    value = os.getenv(name)
    return float(value) if value else default


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "")

    docs_dir: str = os.getenv("DOCS_DIR", "")
    chroma_dir: str = os.getenv("CHROMA_DIR", "")

    logging_dir: str = os.getenv("LOGGING_DIR", "")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "")
    reranker_model: str = os.getenv("RERANKER_MODEL", "")

    chunk_size: int = env_int("CHUNK_SIZE", 900)
    chunk_overlap: int = env_int("CHUNK_OVERLAP", 150)  

    rerank_top_k: int = env_int("RERANK_TOP_K", 40)    



settings = Settings()
    

