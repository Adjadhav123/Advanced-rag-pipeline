import json 
import re 

from collections import defaultdict
from functools import lru_cache
from pathlib import Path 

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from app.config import settings
from app.services.vector_db import get_vector_store
from app.logger import configure_logging

logger = configure_logging()


class QueryVariants(BaseModel):
    queries: list[str] = Field(
        description="Alternative concise search queries."
    )


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    return ChatGroq(
        model=settings.groq_model,
        groq_api_key=settings.groq_api_key,
        temperature=0.1,
        max_tokens=250
    )


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load the reranker model."""
    logger.info(f"Loading reranker model {settings.reranker_model}")

    return CrossEncoder(
        settings.reranker_model,
        device="cpu"
    )


def _history_text(history: list[dict] | None) -> str:
    """Convert chat history to readable text."""
    if not history:
        return ""
    recent = history[-6*2:]
    lines = []

    for msg in recent:
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", "")).strip()

        if content:
            lines.append(
                f"{role}: {content[:3000]}"
            )

    return "\n".join(lines)
        
#Rewriting turns a context-dependent question into a standalone query.
def rewrite_query(question: str, history: list[dict]) -> str:
    """Rewrite question to include context from history."""

    history_text = _history_text(history)

    if not history_text:
        return question.strip()


    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the latest user question as a standalone search query. "
                "Use the conversation only to resolve references such as "
                "'it', 'that', or 'the previous method'. Do not answer the "
                "question. Return only the rewritten query.",
            ),
            "human",
            "Conversation:\n {history}\n\nLatest question:\n{question}",
        ]
    )

    try:

        response = (prompt | get_llm()).invoke(
            {
                "history": history_text,
                "question": question
            }
        )

        rewritten = response.content

        if isinstance(rewritten, str) or rewritten.strip():
            return rewritten.strip()
    
    except Exception as e:
        logger.error(f"Failed to rewrite query: {e}")
        return question.strip()

# Expansion gives your retrieval system multiple ways to discover the same evidence.
def expand_queries(question: str) -> str:
    """Expand the query into multiple variants."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                 "system",
                "Create alternative search queries for a document retrieval "
                "system. Preserve the user's intent. Include useful synonyms "
                "or terminology. Do not answer the question. Return distinct "
                "queries, not explanations.",
            ),
            (
                "human",
                "{question}"
            ),
        ]
    )

    try:
        structured_llm = get_llm().with_structured_output(QueryVariants)
        response = (prompt | structured_llm).invoke(
            {"question": question}
        )

        variants = []
        for query in response.queries:
            clean = query.strip()
            if clean and clean.casefold() != question.casefold():
                variants.append(clean)
        
        unique = []
        seen = {question.casefold()}

        for query in variants:
            if query.casefold() not in seen:
                seen.add(query.casefold())
                unique.append(query)

        return [question] + unique[:6]

    except Exception as e:
        logger.error(f"Failed to expand queries: {e}")
        return [question]

@lru_cache(maxsize=1)
def get_bm25_retriever() -> BM25Retriever:
    """Load Bm25 from the local snapshot."""

    path = Path(settings.bm25_snapshot_path)

    if not path.exists():
        raise FileNotFoundError(
            f"BM25 snapshot not found: {path}. Please ingest/index documents first (e.g. run 'python ingest.py')."
        )

    records = json.loads(path.read_text(encoding="utf-8"))

    if not records:
        raise ValueError(
            f"BM25 snapshot is empty:{path}"
        )

    documents = []

    for record in records:
        metadata = dict(record.get("metadata", {}))
        metadata["chunk_id"] = record["id"]

        documents.append(
            Document(
                page_content=record.get("text",""),
                metadata=metadata   
            )
        )

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = settings.bm25_top_k

    return retriever
        

def _document_id(document: Document) ->str:
    chunk_id = document.metadata.get("chunk_id")

    if chunk_id:
        return str(chunk_id)

    import hashlib

    key = (
        str(document.metadata.get("source", "")) +
        "|"
        + str(document.metadata.get("page","")) +
        "|"
        + document.page_content
    )

    return hashlib.sha256(key.encode("utf-8")).hexdigest()

#RRF combines the rankings from different search methods to improve the chances of retrieving the most relevant documents for an LLM.
def reciprocal_rank_fusion(
    rankings: list[list[Document]],
    k: int = 60
) -> list[Document]:
    scores: dict[str, float] = defaultdict(float)
    documents_by_id : dict[str, Document] = {}

    for ranking in rankings:
        seen_in_ranking = set()

        for rank, doc in enumerate(ranking, start=1):
            doc_id = _document_id(doc)

            if doc_id in seen_in_ranking:
                continue

            seen_in_ranking.add(doc_id)
            scores[doc_id] += 1.0 / (rank + k)
            documents_by_id.setdefault(doc_id, doc)

    ordered_ids = sorted(
        scores,
        key=lambda doc_id: scores[doc_id],
        reverse=True
    )

    fused = []
    for doc_id in ordered_ids:
        doc = documents_by_id[doc_id]
        metadata = dict(doc.metadata)
        metadata["rrf_score"] = scores[doc_id]

        fused.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata
            )
        )

    return fused

def retrieve_candidates(query: str)->list[Document]:
    """Retrieve candidate documents using BM25 and vector search."""

    bm25 = get_bm25_retriever()
    vector_store = get_vector_store()

    query_variants = expand_queries(query)
    rankings : list[list[Document]] = [] 

    for search_query in query_variants:
        try:
            vector_docs = vector_store.similarity_search(
                search_query,
                k=20
            )
            rankings.append(vector_docs)
        
        except Exception as e:
            logger.exception("Vector retrieval failed for query.")

        
        try:
            lexical_docs = bm25.invoke(search_query)
            rankings.append(lexical_docs)
        except Exception as e:
            logger.exception("BM25 retrieval failed for query.")


    fused = reciprocal_rank_fusion(rankings, k=60)
    return fused[:settings.rerank_top_k]


def rerank_documents(
    query:str,
    documents : list[Document]
)-> list[Document]:
    """Use cross-encoder reranker to improve relevance of retrieved documents"""

    if not documents:
        return []

    model = get_reranker()

    pairs = [
        (query, doc.page_content)
        for doc in documents
    ]

    scores = model.predict(pairs, show_progress_bar=False)

    ranked = sorted(
        zip(documents, scores),
        key=lambda pair: float(pair[1]),
        reverse=True
    )

    result = []

    for doc, score in ranked[:settings.rerank_top_k]:
        score = float(score)

        if score < 0.3:
            break

        metadata = dict(doc.metadata)
        metadata["rerank_score"] = score

        result.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata
            )
        )

    return result


def retrieve(
    question: str,
    history: list[dict] | None = None
)-> tuple[str, list[Document]]:

    rewritten = rewrite_query(question, history)

    candidates = retrieve_candidates(rewritten)

    context = rerank_documents(rewritten, candidates)

    return rewritten, context