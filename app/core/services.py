# app/core/services.py

import os
import hashlib
import json
from typing import List, Tuple
from dotenv import load_dotenv

import fitz
import requests
import cohere
from groq import Groq
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.core.database import SessionLocal, ProcessedDocument

load_dotenv()

# --- HELPER FUNCTIONS ---


def process_document_from_url(url: str) -> str:
    """Downloads a PDF from a URL and extracts all its text."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with fitz.open(stream=response.content, filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)


def chunk_text(text: str) -> List[str]:
    """Splits a long text into smaller, overlapping chunks."""
    chunks, chunk_size, overlap = [], 1000, 200
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# --- CORE LOGIC FUNCTIONS ---


def process_and_embed_document(
    url: str,
    embedding_model: SentenceTransformer,
    pinecone_index
) -> List[str]:
    """Manages the full document processing pipeline, using pre-loaded models."""
    doc_id = hashlib.sha256(url.encode()).hexdigest()
    db = SessionLocal()
    cached_doc = db.query(ProcessedDocument).filter(
        ProcessedDocument.id == doc_id).first()
    if cached_doc:
        print(f"INFO:     Cache hit for document. Retrieving chunks from DB.")
        db.close()
        return cached_doc.chunks
    db.close()

    print(f"INFO:     Cache miss. Starting full processing for document.")
    text = process_document_from_url(url)
    chunks = chunk_text(text)

    print("INFO:     Creating embeddings in parallel...")
    pool = embedding_model.start_multi_process_pool()
    embeddings = embedding_model.encode(chunks, pool=pool, batch_size=64)
    embedding_model.stop_multi_process_pool(pool)

    vectors = [{"id": f"{doc_id}_{i}", "values": emb.tolist(), "metadata": {"text": c}}
               for i, (c, emb) in enumerate(zip(chunks, embeddings))]

    pinecone_index.upsert(vectors=vectors, namespace=doc_id)
    print("INFO:     Upsert to Pinecone complete.")

    db = SessionLocal()
    db.add(ProcessedDocument(id=doc_id, url=url, chunks=chunks))
    db.commit()
    print("INFO:     Saved document record and chunks to cache.")
    db.close()
    return chunks


def retrieve_and_rerank_context(
    question: str,
    url: str,
    all_chunks: List[str],
    embedding_model: SentenceTransformer,
    pinecone_index,
    cohere_client
) -> str:
    """Performs hybrid search and re-ranks results using pre-loaded models."""
    doc_id = hashlib.sha256(url.encode()).hexdigest()

    # 1. Hybrid Search
    query_embedding = embedding_model.encode([question]).tolist()
    semantic_results = pinecone_index.query(
        vector=query_embedding, top_k=5, include_metadata=True, namespace=doc_id)
    semantic_chunks = [res['metadata']['text']
                       for res in semantic_results['matches']]

    tokenized_corpus = [doc.split(" ") for doc in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    keyword_chunks = bm25.get_top_n(question.split(" "), all_chunks, n=5)

    candidate_chunks = list(dict.fromkeys(semantic_chunks + keyword_chunks))
    print(
        f"DEBUG:    Found {len(candidate_chunks)} unique candidate chunks before re-ranking.")

    # 2. Re-ranking with Cohere
    reranked_results = cohere_client.rerank(
        model="rerank-english-v3.0",
        query=question,
        documents=candidate_chunks,
        top_n=3
    )

    # --- THE FIX: Use the 'index' from the result to look up the original text ---
    final_chunks = []
    for res in reranked_results.results:
        # The 'index' in the result points to the position in our original 'candidate_chunks' list
        final_chunks.append(candidate_chunks[res.index])

    if not final_chunks:
        print(
            f"WARN:     Re-ranking returned no valid document chunks for question: '{question}'")
        return ""

    final_context = " ".join(final_chunks)
    print(f"DEBUG:    Final context character count: {len(final_context)}")
    return final_context


def generate_answer(question: str, context: str) -> str:
    """Generates a single answer with a Groq->Gemini fallback."""
    if not context or not context.strip():
        print("WARN:     Context is empty. Skipping LLM call.")
        return "The information is not available in the provided document."

    system_prompt = (
        "You are an expert Q&A system. Your final output must be a single string answer. "
        "Based ONLY on the provided context, answer the question. "
        "If the answer is not in the context, state 'The information is not available in the provided document.' "
        "At the end of your answer, include a 'Source:' section quoting the exact sentence(s) from the context."
    )
    final_prompt = f"Context:\n---\n{context}\n---\nQuestion: {question}"

    try:
        # --- Primary LLM: Groq ---
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": final_prompt}],
            model="llama3-70b-8192", temperature=0.1
        )
        return chat_completion.choices[0].message.content
    except Exception as groq_error:
        print(f"WARN:     Groq failed: {groq_error}. Trying Gemini fallback.")
        # --- Fallback LLM: Gemini ---
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        gemini_client = genai.GenerativeModel('gemini-1.5-pro-latest')

        response = gemini_client.generate_content(
            f"{system_prompt}\n{final_prompt}",
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        return response.text
