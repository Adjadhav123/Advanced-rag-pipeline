import hashlib
import logging
from pathlib import Path 

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings 
from app.logger import configure_logging

logger = configure_logging()

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def load_documents(directory_path: str | Path ) -> list[Document]:
    """Load all the supported documents from the directory recursively."""
    root = Path(directory_path)

    if not root.exists():
        raise FileNotFoundError(f"Documents directory not found: {root}")

    documents : list[Document] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        logger.info(f"Loading files from : {path}")

        try:

            suffix = path.suffix.lower()

            if suffix == ".pdf":
                loader = PyPDFLoader(str(path)).load()
            
            elif suffix == ".docx":
                loader = Docx2txtLoader(str(path)).load()

            else:
                loader = TextLoader(
                    str(path),
                    autodetect_encoding=True
                ).load()

            for doc in loader:
                doc.page_content = doc.page_content.strip()
                if not doc.page_content:
                    continue

                try:
                    relative_source = str(path.relative_to(root))
                except ValueError:
                    relative_source = path.name

                doc.metadata["source"] = relative_source
                doc.metadata["file_type"] = suffix.lstrip(".")
                documents.append(doc)
        
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            continue
    
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Used to split the documents into chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    result : list[Document] = []

    seen_ids: set[str] = set()

    for idx, chunk in enumerate(chunks):
        content = chunk.page_content.strip()
        if not content:
            continue

        chunk.page_content = content 
        metadata = chunk.metadata

        source = str(metadata.get("source") or metadata.get("sources") or "")
        page = str(metadata.get("page", ""))
        start_index = str(metadata.get("start_index", metadata.get("start_idx", idx)))

        identity = f"{source}|{page}|{start_index}|{content}"
        chunk_id = hashlib.sha256(identity.encode('utf-8')).hexdigest()

        if chunk_id in seen_ids:
            chunk_id = hashlib.sha256(f"{identity}|{idx}".encode('utf-8')).hexdigest()

        seen_ids.add(chunk_id)
        metadata["chunk_id"] = chunk_id
        result.append(chunk)

    return result


                

            


