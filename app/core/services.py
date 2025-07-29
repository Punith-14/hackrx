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
from groq import Groq
from rank_bm25 import BM25Okapi
from app.core.database import SessionLocal, ProcessedDocument

load_dotenv()

# Singleton Classes (EmbeddingModel, PineconeService, LLMService)


class EmbeddingModel:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            print("INFO:     Loading embedding model...")
            cls._instance = SentenceTransformer('all-MiniLM-L6-v2')
            print("INFO:     Embedding model loaded.")
        return cls._instance


class PineconeService:
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
            if cls.INDEX_NAME not in cls._instance.list_indexes().names():
                print(
                    f"INFO:     Creating new Pinecone index: {cls.INDEX_NAME}")
                cls._instance.create_index(name=cls.INDEX_NAME, dimension=384, metric='cosine', spec=ServerlessSpec(
                    cloud='aws', region='us-east-1'))
            cls._index = cls._instance.Index(cls.INDEX_NAME)
            print("INFO:     Pinecone connection successful.")
        return cls._instance

    @classmethod
    def get_index(cls):
        if cls._index is None:
            cls.get_instance()
        return cls._index


class LLMService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            print("INFO:     Initializing Groq client...")
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY must be set.")
            cls._instance = Groq(api_key=api_key)
            print("INFO:     Groq client initialized.")
        return cls._instance

# Core Logic Functions


def process_document_and_upsert(url: str) -> List[str]:
    doc_id = hashlib.sha256(url.encode()).hexdigest()
    db = SessionLocal()
    try:
        if db.query(ProcessedDocument).filter(ProcessedDocument.id == doc_id).first():
            print(
                f"INFO:     Cache hit for document URL: {url}. Re-chunking for context.")
            text = process_document_from_url(url)
            return chunk_text(text)
    finally:
        db.close()

    print(
        f"INFO:     Cache miss for document URL: {url}. Starting full processing...")
    text = process_document_from_url(url)
    chunks = chunk_text(text)
    embedding_model = EmbeddingModel.get_instance()
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    vectors = [{"id": f"{doc_id}_chunk_{i}", "values": embeddings[i].tolist(
    ), "metadata": {"text": chunk}} for i, chunk in enumerate(chunks)]

    index = PineconeService.get_index()
    print(
        f"INFO:     Upserting {len(vectors)} vectors to Pinecone namespace: {doc_id[:8]}...")
    index.upsert(vectors=vectors, namespace=doc_id)
    print("INFO:     Upsert complete.")

    db = SessionLocal()
    try:
        new_doc_record = ProcessedDocument(id=doc_id, url=url)
        db.add(new_doc_record)
        db.commit()
        print(f"INFO:     Saved document record to cache.")
    finally:
        db.close()

    return chunks


def get_context_for_questions(questions: List[str], url: str, all_chunks: List[str]) -> List[Tuple[str, str]]:
    doc_id = hashlib.sha256(url.encode()).hexdigest()
    embedding_model = EmbeddingModel.get_instance()
    index = PineconeService.get_index()
    tokenized_corpus = [doc.split(" ") for doc in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    qa_contexts = []
    for question in questions:
        print(f"INFO:     Hybrid search for question: '{question}'")

        query_embedding = embedding_model.encode([question]).tolist()
        semantic_results = index.query(
            vector=query_embedding, top_k=3, include_metadata=True, namespace=doc_id)
        semantic_chunks = [res['metadata']['text']
                           for res in semantic_results['matches']]

        tokenized_query = question.split(" ")
        keyword_chunks = bm25.get_top_n(tokenized_query, all_chunks, n=3)

        combined_chunks = semantic_chunks + keyword_chunks
        unique_chunks = list(dict.fromkeys(combined_chunks))

        context = " ".join(unique_chunks)
        qa_contexts.append((question, context))

    return qa_contexts


def generate_answers_in_batch(qa_contexts: List[Tuple[str, str]]) -> List[str]:
    llm = LLMService.get_instance()
    system_prompt = (
        "You are an expert Q&A system. Your task is to answer a series of questions based on the provided context for each question."
        "Follow these rules:"
        "1. Answer each question based ONLY on its own provided context."
        "2. If the answer is not in the context, you MUST state 'The information is not available in the provided document.'"
        "3. Your final output MUST be a single JSON object with one key: 'answers'. The value of 'answers' must be a JSON array of strings."
        "4. **IMPORTANT**: At the end of each answer string, you MUST include a 'Source:' section, quoting the exact sentence(s) from the context that justifies your answer."
        "Example output format: {\"answers\": [\"The waiting period is 36 months. Source: 'The policy has a waiting period of thirty-six (36) months from the first policy inception date.'\", \"Answer to question 2.\"]}"
        "Do not include any other text, explanations, or markdown formatting, just the raw JSON object."
    )
    user_prompt_parts = []
    for i, (question, context) in enumerate(qa_contexts):
        user_prompt_parts.append(f"Question {i+1}: {question}")
        user_prompt_parts.append(
            f"Context for Question {i+1}:\n---\n{context}\n---")
    final_user_prompt = "\n".join(user_prompt_parts)
    print("INFO:     Sending a single batch request to the Groq LLM for all questions.")
    try:
        chat_completion = llm.chat.completions.create(messages=[{"role": "system", "content": system_prompt}, {
                                                      "role": "user", "content": final_user_prompt},], model="llama3-8b-8192", temperature=0.0, response_format={"type": "json_object"})
        response_text = chat_completion.choices[0].message.content
        response_data = json.loads(response_text)
        answers = response_data.get("answers", [])
        if len(answers) != len(qa_contexts):
            raise ValueError(
                f"LLM returned {len(answers)} answers for {len(qa_contexts)} questions.")
        return answers
    except Exception as e:
        print(f"Error during batch LLM generation or JSON parsing: {e}")
        return ["Error generating an answer due to an issue with the AI service." for _ in qa_contexts]


def process_document_from_url(url: str) -> str:
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
    if not text:
        return []
    chunks = []
    start_index = 0
    while start_index < len(text):
        end_index = start_index + chunk_size
        chunks.append(text[start_index:end_index])
        start_index += chunk_size - chunk_overlap
    return chunks
