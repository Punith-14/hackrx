# app/main.py

<<<<<<< HEAD
import os
from fastapi import FastAPI
from app.api.models import QueryRequest, QueryResponse
from app.core.services import (
    process_and_embed_document,
    retrieve_and_rerank_context,
    generate_answer
)
from app.core.database import init_db
import concurrent.futures

# --- Import models and clients for global state ---
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import cohere

app = FastAPI(title="HackRx Final Submission")

# This dictionary will hold our initialized models to be shared across threads
app_state = {}


@app.on_event("startup")
def on_startup():
    """
    Initializes all necessary services and models once on application startup
    and stores them in a global state dictionary to be shared safely.
    """
    print("INFO:     Initializing database...")
    init_db()

    print("INFO:     Loading Embedding Model...")
    app_state["embedding_model"] = SentenceTransformer('all-MiniLM-L6-v2')

    print("INFO:     Initializing Pinecone...")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY must be set.")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "hackrx-index"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name, dimension=384, metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    app_state["pinecone_index"] = pc.Index(index_name)

    print("INFO:     Initializing Cohere Client...")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY must be set.")
    app_state["cohere_client"] = cohere.Client(cohere_api_key)

    print("INFO:     Application startup complete.")


def process_single_question(args):
    """
    Helper function for parallel processing. It correctly unpacks all arguments
    including the shared models dictionary.
    """
    question, url, all_chunks, models = args

    print(f"INFO:     Processing question: '{question}'")
    context = retrieve_and_rerank_context(
        question, url, all_chunks,
        embedding_model=models["embedding_model"],
        pinecone_index=models["pinecone_index"],
        cohere_client=models["cohere_client"]
    )
    answer = generate_answer(question, context)
    return answer
=======
from fastapi import FastAPI, HTTPException, status
from app.api.models import QueryRequest, QueryResponse
from app.core.services import (
    EmbeddingModel, PineconeService, LLMService,
    process_document_and_upsert, get_context_for_questions, generate_answers_in_batch
)
from app.core.database import init_db

# =============================================================================
# Application Initialization
# =============================================================================

app = FastAPI(
    title="HackRx Intelligent Query-Retrieval System",
    description="An advanced RAG system to answer questions from documents.",
    version="1.0.0"
)

# =============================================================================
# Application Startup Event
# =============================================================================


@app.on_event("startup")
async def startup_event():
    """
    Initializes all necessary services and database tables
    when the application starts.
    """
    init_db()
    EmbeddingModel.get_instance()
    PineconeService.get_instance()
    LLMService.get_instance()

# =============================================================================
# API Endpoints
# =============================================================================
>>>>>>> be351b9035f8c6dfc120573b5e18b7d0c60dff92


@app.get("/", tags=["Health Check"])
async def root():
<<<<<<< HEAD
    """A simple health check endpoint."""
    return {"status": "ok", "message": "API is running!"}


@app.post("/hackrx/run", response_model=QueryResponse, tags=["Query System"])
async def run_query_pipeline(request: QueryRequest):
    """
    Main API endpoint orchestrating the full RAG pipeline with parallel processing.
    """
    try:
        all_chunks = process_and_embed_document(
            url=str(request.documents),
            embedding_model=app_state["embedding_model"],
            pinecone_index=app_state["pinecone_index"]
        )

        # Each task now correctly includes a reference to the shared app_state dictionary
        tasks = [(q, str(request.documents), all_chunks, app_state)
                 for q in request.questions]

        final_answers = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(process_single_question, tasks)
            final_answers = list(results)

        return QueryResponse(answers=final_answers)
    except Exception as e:
        print(f"CRITICAL: An error occurred in the main pipeline: {e}")
        error_response = [
            f"An unexpected server error occurred." for _ in request.questions]
        return QueryResponse(answers=error_response)
=======
    """
    A simple health check endpoint to confirm the API is running.
    """
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
    This is the main entry point for the application's logic.
    """
    try:
        # Step 1: Process document (uses caching) and get text chunks
        all_chunks = process_document_and_upsert(str(request.documents))

        # Step 2: Retrieve context for all questions using hybrid search
        qa_contexts = get_context_for_questions(
            request.questions,
            str(request.documents),
            all_chunks
        )

        # Step 3: Generate all answers in a single, efficient batch call
        final_answers = generate_answers_in_batch(qa_contexts)

        return QueryResponse(answers=final_answers)

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred: {str(e)}"
        )
>>>>>>> be351b9035f8c6dfc120573b5e18b7d0c60dff92
