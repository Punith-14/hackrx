# 🚀 HackRx: High-Performance Intelligent Query-Retrieval System

An advanced, production-grade **Retrieval-Augmented Generation (RAG)** system built for the **HackRx hackathon**. Designed for **maximum accuracy, speed, and reliability**, this system can process large PDF documents and answer complex natural language questions with **verifiable, source-backed answers**.

---

## ✨ Features

### 🔍 High-Accuracy Retrieval Pipeline
- **Hybrid Search**: Combines keyword search (BM25) and semantic search (Pinecone) for broad and deep document understanding.
- **Cohere Re-ranking**: Uses Cohere’s state-of-the-art model to re-rank results from hybrid search for clean and precise context selection.

### ⚡ Optimized for Speed & Performance
- **Multithreaded Query Execution**: Processes all questions concurrently to minimize total response time.
- **Parallel Embedding Creation**: Accelerates document embedding by using all CPU cores.
- **Smart Caching with PostgreSQL**: Skips reprocessing for already-seen documents by storing parsed chunks.

### 🧠 Enhanced Reliability & Explainability
- **LLM Fallback System**: First uses Groq’s LLaMA 3 70B for fast answers, then falls back to Gemini 1.5 Pro if needed.
- **Direct Source Citations**: Every answer includes quoted source text from the document for transparency and trust.

---

## 🛠️ Tech Stack

| Component        | Technology                       |
|------------------|-----------------------------------|
| Backend          | FastAPI                          |
| Vector Database  | Pinecone                         |
| Re-ranking       | Cohere API                       |
| Primary LLM      | Groq (LLaMA 3 70B)               |
| Fallback LLM     | Google Gemini 1.5 Pro            |
| Cache DB         | PostgreSQL (via Neon)            |
| Libraries        | Sentence-Transformers, PyMuPDF, Rank-BM25, SQLAlchemy |

---

## 📦 Getting Started

### ✅ Prerequisites
- Python **3.10+**
- API keys for:
  - Pinecone
  - Cohere
  - Groq
  - Google AI
- PostgreSQL connection string (e.g., from [Neon](https://neon.tech))

---

### 📥 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd hackrx-final
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    Create a `.env` file in the root directory and add your credentials:
    ```
    PINECONE_API_KEY="your_pinecone_api_key"
    COHERE_API_KEY="your_cohere_api_key"
    GROQ_API_KEY="your_groq_api_key"
    GOOGLE_API_KEY="your_google_api_key"
    DATABASE_URL="your_postgresql_connection_string"
    ```

### Running the Application

To start the local development server, run:
    ```bash
    uvicorn app.main:app --reload  # The API will be available at http://127.0.0.1:8000
    ```
### ⚙️ API Usage

The application has one main endpoint for processing documents and questions.
- **Endpoint:** POST /hackrx/run
- **Request Body:**
    ```
    JSON
    {
    "documents": "URL_TO_YOUR_PDF_DOCUMENT",
    "questions": [
        "Your first question?",
        "Your second question?"
    ]
    }
    ```
- **Success Response:**
    ```
    JSON
    {
    "answers": [
        "Answer to your first question. Source: 'The source sentence from the document.'",
        "Answer to your second question. Source: '...'"
    ]
    }
     ```
