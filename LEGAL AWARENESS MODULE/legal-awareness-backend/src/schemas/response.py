from pydantic import BaseModel
from typing import List, Optional

class Answer(BaseModel):
    question: str
    answer: str
    confidence: Optional[float] = None

class ResponseModel(BaseModel):
    success: bool
    message: str
    data: Optional[List[Answer]] = None

class ErrorResponseModel(BaseModel):
    success: bool
    error: str
    code: Optional[int] = None