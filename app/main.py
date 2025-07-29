# app/main.py

from fastapi import FastAPI, HTTPException, status
from app.api.models import QueryRequest, QueryResponse
from app.core.services import (
    EmbeddingModel, PineconeService, LLMService,
    process_document_and_upsert, get_context_for_questions, generate_answers_in_batch
)
from app.core.database import init_db

app = FastAPI(
    title="HackRx Intelligent Query-Retrieval System",
    description="An advanced RAG system to answer questions from documents.",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Load models, initialize services, and create DB tables on startup."""
    init_db()
    EmbeddingModel.get_instance()
    PineconeService.get_instance()
    LLMService.get_instance()


@app.get("/", tags=["Health Check"])
async def root():
    return {"status": "ok", "message": "API is running!"}


@app.post(
    "/hackrx/run",
    response_model=QueryResponse,
    tags=["Query System"],
    summary="Process a document and generate answers",
)
async def run_query(request: QueryRequest):
    """
    Orchestrates the full RAG pipeline with caching and hybrid search.
    """
    try:
        all_chunks = process_document_and_upsert(str(request.documents))
        qa_contexts = get_context_for_questions(
            request.questions,
            str(request.documents),
            all_chunks
        )
        final_answers = generate_answers_in_batch(qa_contexts)
        return QueryResponse(answers=final_answers)

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred: {str(e)}"
        )
