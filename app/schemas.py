from typing import Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role : Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=12000)
    session_id: str | None = Field(default=None, max_length=128)
    chat_history : list[ChatMessage] = Field(default_factory=list)


class SourceItem(BaseModel):
    source: str 
    page: int | None = None 
    chunk_id: str 
    rerank_score: float

class ChatResponse(BaseModel):
    answer: str 
    sources: list[SourceItem]
    request_id: str
    