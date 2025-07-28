# app/api/models.py

from pydantic import BaseModel, Field, HttpUrl
from typing import List

# =============================================================================
# API Request Models
# =============================================================================


class QueryRequest(BaseModel):
    """
    Defines the structure for the incoming POST request to the /hackrx/run endpoint.
    It expects a 'documents' URL and a list of 'questions'.
    """
    documents: HttpUrl = Field(
        ...,
        description="A valid and accessible URL pointing to the document to be processed."
    )
    questions: List[str] = Field(
        ...,
        min_length=1,
        description="A list of one or more questions to be answered based on the document."
    )

# =============================================================================
# API Response Models
# =============================================================================


class QueryResponse(BaseModel):
    """
    Defines the structure for the outgoing JSON response.
    It provides a list of string answers, corresponding to the order of questions.
    """
    answers: List[str] = Field(
        ...,
        description="A list of generated answers. Each answer corresponds to a question in the request."
    )
