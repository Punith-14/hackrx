# app/api/models.py

from pydantic import BaseModel, Field, HttpUrl
from typing import List


class QueryRequest(BaseModel):
    """Defines the structure for the incoming API request."""
    documents: HttpUrl = Field(...,
                               description="A URL to the document to be processed.")
    questions: List[str] = Field(..., min_length=1,
                                 description="A list of questions about the document.")


class QueryResponse(BaseModel):
    """Defines the structure for the API response."""
    answers: List[str] = Field(..., description="A list of generated answers.")
