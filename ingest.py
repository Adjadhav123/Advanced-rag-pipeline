import argparse
import json
import logging

from app.config import settings
from app.logger import configure_logging
from app.services.index_service import index_documents


def main() -> None:
    configure_logging()
    settings.validate()

    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG indexes."
    )
    parser.add_argument(
        "--directory",
        default=settings.docs_dir,
        help="Directory containing TXT, MD, PDF, or DOCX files.",
    )
    args = parser.parse_args()

    count = index_documents(args.directory)
    print(f"Successfully indexed {count} document chunks into Chroma and BM25 snapshot.")


if __name__ == "__main__":
    main()