# HackRx: Intelligent Query-Retrieval System

This project is an advanced Retrieval-Augmented Generation (RAG) system built for the HackRx hackathon. It can process large documents (PDFs) and accurately answer natural language questions based on their content.

## ✨ Features

This system goes beyond a basic RAG implementation and includes several advanced features for improved performance, accuracy, and explainability:

-   **Intelligent Document Caching:** Uses a PostgreSQL database to cache processed documents, dramatically improving latency on subsequent requests for the same document.
-   **Direct Citations:** The LLM is prompted to include the specific source sentence from the document in its answer, providing clear explainability.
-   **Hybrid Search:** Combines traditional keyword search (BM25) with modern semantic search (Pinecone) to ensure the most relevant context is retrieved.
-   **Context Re-ranking:** Employs a Cross-Encoder model to re-rank the retrieved context, filtering out noise and providing the final LLM with only the most precise information.
-   **Modular, Singleton-Based Architecture:** Services for embedding, database connections, and LLM calls are managed as singletons for performance and clean code.

## 🛠️ Tech Stack

-   **Backend:** FastAPI
-   **Vector Database:** Pinecone
-   **LLM Service:** Groq (running Llama 3)
-   **Cache Database:** PostgreSQL (via Neon)
-   **Core Libraries:** Sentence-Transformers, PyMuPDF, Rank-BM25, SQLAlchemy, Psycopg2

## 🚀 Getting Started

### Prerequisites

-   Python 3.10+
-   API keys for Pinecone and Groq
-   A PostgreSQL database connection string (e.g., from Neon)

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd hackrx-intelligent-query
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
    PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
    GROQ_API_KEY="YOUR_GROQ_API_KEY"
    DATABASE_URL="YOUR_POSTGRESQL_CONNECTION_STRING"
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
    JSON
    {
    "answers": [
        "Answer to your first question. Source: 'The source sentence from the document.'",
        "Answer to your second question. Source: '...'"
    ]
    }

