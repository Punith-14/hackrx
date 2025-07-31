# app/core/services.py

import os
import requests
import fitz
import json
import hashlib
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from app.core.database import SessionLocal, ProcessedDocument
from rank_bm25 import BM25Okapi

# Load environment variables from the .env file
load_dotenv()

# =============================================================================
# Singleton Classes for Services
# =============================================================================


class EmbeddingModel:
    """Singleton for the SentenceTransformer model to ensure it's loaded only once."""
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            print("INFO:     Loading embedding model...")
            cls._instance = SentenceTransformer('all-MiniLM-L6-v2')
            print("INFO:     Embedding model loaded.")
        return cls._instance


class PineconeService:
    """Singleton for the Pinecone service to manage connection and index."""
    _instance = None
    _index = None
    INDEX_NAME = "hackrx-index"

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            print("INFO:     Initializing Pinecone connection...")
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY must be set.")

            cls._instance = Pinecone(api_key=api_key)
            # Create index if it doesn't exist
            if cls.INDEX_NAME not in cls._instance.list_indexes().names():
                print(
                    f"INFO:     Creating new Pinecone index: {cls.INDEX_NAME}")
                cls._instance.create_index(
                    name=cls.INDEX_NAME, dimension=384, metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
            cls._index = cls._instance.Index(cls.INDEX_NAME)
            print("INFO:     Pinecone connection successful.")
        return cls._instance

    @classmethod
    def get_index(cls):
        if cls._index is None:
            cls.get_instance()
        return cls._index


class LLMService:
    """Singleton for the Generative AI Model (Gemini)."""
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            print("INFO:     Initializing Generative AI model (Gemini 2.5 Flash)...")
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY must be set.")
            genai.configure(api_key=api_key)
            cls._instance = genai.GenerativeModel('gemini-1.5-flash-latest')
            print("INFO:     Generative AI model initialized.")
        return cls._instance

# =============================================================================
# Core Logic Functions
# =============================================================================


def process_document_and_upsert(url: str) -> List[str]:
    """
    Main document processing function. Checks cache first. If a new document,
    it downloads, chunks, embeds, and upserts to Pinecone, then caches the
    result in PostgreSQL.

    Args:
        url (str): The URL of the document to process.

    Returns:
        List[str]: A list of the document's text chunks.
    """
    doc_id = hashlib.sha256(url.encode()).hexdigest()
    db = SessionLocal()
    try:
        # Check PostgreSQL cache first
        cached_doc = db.query(ProcessedDocument).filter(
            ProcessedDocument.id == doc_id).first()
        if cached_doc:
            print(
                f"INFO:     Cache hit for document URL: {url}. Retrieving chunks from DB.")
            return cached_doc.chunks
    finally:
        db.close()

    # --- Cache Miss: Full processing pipeline ---
    print(
        f"INFO:     Cache miss for document URL: {url}. Starting full processing...")
    text = process_document_from_url(url)
    chunks = chunk_text(text)
    embedding_model = EmbeddingModel.get_instance()
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    vectors = [{"id": f"{doc_id}_chunk_{i}", "values": embeddings[i].tolist(
    ), "metadata": {"text": chunk}} for i, chunk in enumerate(chunks)]

    # Upsert vectors to Pinecone in a unique namespace for this document
    index = PineconeService.get_index()
    print(
        f"INFO:     Upserting {len(vectors)} vectors to Pinecone namespace: {doc_id[:8]}...")
    index.upsert(vectors=vectors, namespace=doc_id)
    print("INFO:     Upsert complete.")

    # Save record and chunks to PostgreSQL cache
    db = SessionLocal()
    try:
        new_doc_record = ProcessedDocument(id=doc_id, url=url, chunks=chunks)
        db.add(new_doc_record)
        db.commit()
        print(f"INFO:     Saved document record and chunks to cache.")
    finally:
        db.close()

    return chunks


def get_context_for_questions(questions: List[str], url: str, all_chunks: List[str]) -> List[Tuple[str, str]]:
    """
    For a list of questions, performs hybrid search to find the best context for each.

    Args:
        questions (List[str]): The list of user questions.
        url (str): The URL of the source document to create a namespace.
        all_chunks (List[str]): All text chunks from the document for keyword search.

    Returns:
        List[Tuple[str, str]]: A list of tuples, each containing a question and its retrieved context.
    """
    doc_id = hashlib.sha256(url.encode()).hexdigest()
    embedding_model = EmbeddingModel.get_instance()
    index = PineconeService.get_index()

    # Initialize keyword search model
    tokenized_corpus = [doc.split(" ") for doc in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    qa_contexts = []
    for question in questions:
        print(f"INFO:     Hybrid search for question: '{question}'")

        # 1. Semantic search via Pinecone
        query_embedding = embedding_model.encode([question]).tolist()
        semantic_results = index.query(
            vector=query_embedding, top_k=3, include_metadata=True, namespace=doc_id)
        semantic_chunks = [res['metadata']['text']
                           for res in semantic_results['matches']]

        # 2. Keyword search via BM25
        tokenized_query = question.split(" ")
        keyword_chunks = bm25.get_top_n(tokenized_query, all_chunks, n=3)

        # 3. Combine and deduplicate results
        combined_chunks = semantic_chunks + keyword_chunks
        unique_chunks = list(dict.fromkeys(combined_chunks))

        context = " ".join(unique_chunks)
        qa_contexts.append((question, context))

    return qa_contexts


def generate_answers_in_batch(qa_contexts: List[Tuple[str, str]]) -> List[str]:
    """
    Generates answers for all questions in a single, efficient batch call to the LLM.

    Args:
        qa_contexts (List[Tuple[str, str]]): A list of (question, context) pairs.

    Returns:
        List[str]: A list of answers from the LLM.
    """
    llm = LLMService.get_instance()

    # A detailed system prompt to guide the LLM's behavior and output format
    system_prompt = (
        "You are an expert Q&A system. Your task is to answer a series of questions based on the provided context for each question."
        "Follow these rules:"
        "1. Answer each question based ONLY on its own provided context."
        "2. If the answer is not in the context, you MUST state 'The information is not available in the provided document.'."
        "3. Your final output MUST be a single JSON object with one key: 'answers'. The value of 'answers' must be a JSON array of strings."
        "4. Do not include any introductory text, explanations, or markdown formatting like ```json. Your entire response must be only the raw JSON object."
        "Example output format: {\"answers\": [\"Answer to question 1.\", \"Answer to question 2.\"]}"
    )

    # Combine all questions and contexts into one large prompt
    user_prompt_parts = []
    for i, (question, context) in enumerate(qa_contexts):
        user_prompt_parts.append(f"Question {i+1}: {question}")
        user_prompt_parts.append(
            f"Context for Question {i+1}:\n---\n{context}\n---")

    final_prompt = system_prompt + "\n" + "\n".join(user_prompt_parts)
    print("INFO:     Sending a single batch request to the Gemini LLM for all questions.")

    try:
        # Call the LLM with relaxed safety settings to prevent false positives
        response = llm.generate_content(
            final_prompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

        print(f"DEBUG:    Raw LLM response text: {response.text}")
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")

        if not cleaned_response:
            raise ValueError("LLM returned an empty response after cleaning.")

        # Parse the JSON object and extract the 'answers' list
        response_data = json.loads(cleaned_response)
        answers = response_data.get("answers", [])
        return answers

    except Exception as e:
        print(f"Error during batch LLM generation or JSON parsing: {e}")
        return ["Error generating an answer due to an issue with the AI service." for _ in qa_contexts]


def process_document_from_url(url: str) -> str:
    """Helper function to download a PDF and extract its text."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        pdf_content = response.content
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        full_text = "".join([page.get_text() for page in doc])
        doc.close()
        return full_text
    except Exception as e:
        raise Exception(f"Failed to process document from URL: {e}") from e


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Helper function to split text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    start_index = 0
    while start_index < len(text):
        end_index = start_index + chunk_size
        chunks.append(text[start_index:end_index])
        start_index += chunk_size - chunk_overlap
    return chunks
