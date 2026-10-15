
# Production RAG

A document-grounded RAG application built with LangChain, Chroma,
BM25, Reciprocal Rank Fusion, CrossEncoder reranking, Groq, and FastAPI.

## Features

- TXT, Markdown, PDF, and DOCX ingestion
- Recursive text splitting with chunk metadata
- Persistent Chroma vector database
- BM25 keyword retrieval
- Multi-query expansion
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Grounded responses with source citations
- CLI and FastAPI interface
- Optional API-key authentication
- Basic unit tests

## Requirements

- Python 3.11 recommended
- A Groq API key
- Internet access for initial model downloads and Groq API calls

## Windows setup

Create and activate a virtual environment:

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1

Install dependencies:

    python -m pip install --upgrade pip
    pip install -r requirements.txt

Create the environment file:

    Copy-Item .env.example .env

Edit `.env` and set `GROQ_API_KEY`.

Add your documents to the `docs/` folder. Supported formats:

- .txt
- .md
- .pdf
- .docx

Build the indexes:

    python ingest.py

Start the CLI:

    python chat_cli.py

Start the API:

    uvicorn api:app --host 127.0.0.1 --port 8000

Open the interactive API documentation:

    http://127.0.0.1:8000/docs

## Example API request

POST /ask

    {
      "question": "What are the main findings in the documents?",
      "session_id": "demo-session",
      "history": []
    }

Example PowerShell request:

    $body = @{
        question = "What are the main findings in the documents?"
        session_id = "demo-session"
        history = @()
    } | ConvertTo-Json -Depth 5

    Invoke-RestMethod `
        -Uri "http://127.0.0.1:8000/ask" `
        -Method Post `
        -ContentType "application/json" `
        -Body $body

If RAG_API_KEY is configured, include the X-API-Key header:

    $headers = @{ "X-API-Key" = "your-configured-key" }

## Reindexing

Run `python ingest.py` after adding or updating documents.

The BM25 snapshot is regenerated from the documents present in Chroma.
This starter does not automatically reconcile deleted source files against
the vector database. For reliable updates and deletions, maintain a source
manifest and explicitly remove stale chunk IDs.

## Important production considerations

- Protect the API with authentication, TLS, rate limits, and request limits.
- Do not expose the development server directly to the public internet.
- Configure access controls if different users can access different documents.
- Validate citations against retrieved source labels.
- Add an evaluation dataset for retrieval recall, answer faithfulness,
  citation accuracy, latency, and cost.
- Add structured monitoring, retries, timeouts, and deployment health checks.
- Review dependency versions and test the complete pipeline in your environment.
- BM25 and Chroma must represent the same indexed corpus.
- Cross-encoder scores are ranking scores, not calibrated probabilities.
- The provided `session_id` is not a persistent server-side chat session.
  Conversation history is supplied by the client.
- This is a production-oriented starter, not a fully hardened production
  deployment.
