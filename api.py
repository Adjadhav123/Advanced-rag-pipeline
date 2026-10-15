from app.schemas import ChatRequest
import logging
import secrets
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from app.config import settings
from app.logger import configure_logging
from app.schemas import ChatMessage, ChatResponse, SourceItem
from app.services.answer_services import generate_answer
from app.services.retrieval_services import retrieve

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Production RAG API",
    description=(
        "Document-grounded question answering using hybrid retrieval, "
        "RRF, cross-encoder reranking, and Groq."
    ),
    version="1.0.0",
)


def verify_api_key(
    x_api_key: str | None = Header(default=None),
) -> None:
    if not settings.api_key:
        return

    if not x_api_key or not secrets.compare_digest(
        x_api_key, settings.api_key
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/ask",
    response_model=ChatResponse,
    dependencies=[Depends(verify_api_key)],
)
def ask(request: ChatRequest):
    request_id = str(uuid.uuid4())

    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    history = [
        item.model_dump()
        for item in request.history[-settings.max_history_turns * 2:]
    ]

    try:
        _, documents = retrieve(question, history)
        answer, source_data = generate_answer(
            question=question,
            history=history,
            documents=documents,
        )

        sources = [
            SourceItem(
                source=item["source"],
                page=item["page"],
                chunk_id=item["chunk_id"],
                rerank_score=item["rerank_score"],
            )
            for item in source_data
        ]

        logger.info(
            "RAG request completed request_id=%s sources=%d",
            request_id,
            len(sources),
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            request_id=request_id,
        )

    except FileNotFoundError as exc:
        logger.exception("Index is not ready.")
        raise HTTPException(
            status_code=503,
            detail="The retrieval index is not ready. Run ingestion first.",
        ) from exc

    except Exception as exc:
        logger.exception("RAG request failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process request. Request ID: {request_id}",
        ) from exc