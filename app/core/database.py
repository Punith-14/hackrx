# app/core/database.py

import os
from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

<<<<<<< HEAD
=======
# Get the database URL from environment variables
>>>>>>> be351b9035f8c6dfc120573b5e18b7d0c60dff92
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL must be set in the .env file")

<<<<<<< HEAD
=======
# SQLAlchemy setup for database connection
>>>>>>> be351b9035f8c6dfc120573b5e18b7d0c60dff92
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

<<<<<<< HEAD

class ProcessedDocument(Base):
    """A table to cache processed documents and their text chunks."""
    __tablename__ = "processed_documents"
=======
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

>>>>>>> be351b9035f8c6dfc120573b5e18b7d0c60dff92
    id = Column(String, primary_key=True, index=True)
    url = Column(String, unique=True, index=True)
    processed_at = Column(DateTime, default=datetime.utcnow)
    chunks = Column(JSONB)

<<<<<<< HEAD

def init_db():
    """Creates database tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
=======
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
>>>>>>> be351b9035f8c6dfc120573b5e18b7d0c60dff92
