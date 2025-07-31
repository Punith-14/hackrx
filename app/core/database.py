# app/core/database.py

import os
from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Get the database URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL must be set in the .env file")

# SQLAlchemy setup for database connection
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# =============================================================================
# Database Models (Tables)
# =============================================================================


class ProcessedDocument(Base):
    """
    A table to store a record of documents that have already been processed
    and embedded into Pinecone, acting as our cache. It also stores the
    text chunks to avoid re-processing on cache hits.
    """
    __tablename__ = "processed_documents"

    id = Column(String, primary_key=True, index=True)
    url = Column(String, unique=True, index=True)
    processed_at = Column(DateTime, default=datetime.utcnow)
    chunks = Column(JSONB)

# =============================================================================
# Database Initialization
# =============================================================================


def init_db():
    """
    Creates all database tables defined in the Base metadata.
    This is called once on application startup.
    """
    print("INFO:     Initializing database and creating tables...")
    Base.metadata.create_all(bind=engine)
    print("INFO:     Database initialization complete.")
