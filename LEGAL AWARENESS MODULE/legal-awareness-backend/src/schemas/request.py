from pydantic import BaseModel
from typing import List, Optional

class QuestionRequest(BaseModel):
    question: str

class AnswerRequest(BaseModel):
    question: str
    answer: str

class QueryRequest(BaseModel):
    user_query: str
    top_k: Optional[int] = 1
    threshold: Optional[float] = 70.0

class BulkQueryRequest(BaseModel):
    queries: List[QueryRequest]