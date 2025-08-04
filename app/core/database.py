# app/core/database.py

import os
from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL must be set in the .env file")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ProcessedDocument(Base):
    """A table to cache processed documents and their text chunks."""
    __tablename__ = "processed_documents"
    id = Column(String, primary_key=True, index=True)
    url = Column(String, unique=True, index=True)
    processed_at = Column(DateTime, default=datetime.utcnow)
    chunks = Column(JSONB)


def init_db():
    """Creates database tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
