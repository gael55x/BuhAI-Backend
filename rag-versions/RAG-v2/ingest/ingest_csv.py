import os
import sys
import pandas as pd
import chromadb
import google.generativeai as genai
from sqlalchemy.orm import Session
import logging
from dotenv import load_dotenv
from typing import List

# Load environment variables from .env file
load_dotenv()

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.models import (
    init_db,
    CGMStream,
    MealEvent,
    ActivityLog,
    SleepLog,
    CGMAggregate,
)

# --- CONFIGURATION ---
DB_PATH = "data/buhai.db"
CSV_DIR = "data/dataset-user/"
VECTOR_STORE_DIR = "vector_store"
CHROMA_COLLECTION = "buhai_rag_collection"
# We will now use the Gemini API key.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- EMBEDDING FUNCTION ---
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.error("GEMINI_API_KEY not found in .env file. Ingestion will fail.")
    sys.exit(1)

def embed_documents(documents: List[str]):
    """Embeds a list of documents using the Gemini API."""
    try:
        # Note: The free tier for embedding has a limit of 100 documents per call.
        # For larger datasets, this would need to be batched.
        result = genai.embed_content(
            model="models/embedding-001",
            content=documents,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Failed to embed documents: {e}")
        return []

# --- HELPER FUNCTIONS ---
def get_db_session():
    """Initializes and returns a database session."""
    SessionLocal = init_db(DB_PATH)
    return SessionLocal()

def get_chroma_client():
    """Initializes and returns a ChromaDB client and collection."""
    client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION
    )
    return collection

def clean_column_names(df):
    """Clean DataFrame column names to be valid Python identifiers."""
    df.columns = [col.lower().replace('+', '_').replace('-', '_') for col in df.columns]
    return df

def load_and_clean_csv(file_path: str, date_columns: list):
    """Loads a CSV, coercing date columns and cleaning names."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    df = pd.read_csv(file_path)
    for col in date_columns:
        if col in df.columns:
            # Use the original column name for conversion
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Clean names *after* processing columns by original name
    df = clean_column_names(df)
    
    # Replace all numpy-style Not-a-Time/Number with Python's None.
    # This is critical for SQLAlchemy to be able to insert NULL values.
    df = df.astype(object).where(pd.notnull(df), None)
    return df
    
# --- DATA INGESTION ---

def ingest_cgm_stream(session: Session, file_path: str):
    logger.info(f"Ingesting CGM stream from {file_path}...")
    df = load_and_clean_csv(file_path, date_columns=['timestamp'])
    if df is None: return

    for _, row in df.iterrows():
        # The unique key 'timestamp' must be valid
        if row['timestamp'] is None:
            continue
        exists = session.query(CGMStream).filter_by(timestamp=row['timestamp']).first()
        if not exists:
            record = CGMStream(**row.to_dict())
            session.add(record)
    session.commit()
    logger.info("CGM stream ingestion complete.")

def ingest_meal_events(session: Session, collection, file_path: str):
    logger.info(f"Ingesting meal events from {file_path}...")
    df = load_and_clean_csv(file_path, date_columns=['timestamp', 'date'])
    if df is None: return

    documents, metadatas, ids = [], [], []
    
    for _, row in df.iterrows():
        if row['timestamp'] is None:
            continue
        exists = session.query(MealEvent).filter_by(timestamp=row['timestamp']).first()
        if not exists:
            record_data = row.to_dict()
            # Convert timestamp to python date object if it exists
            if record_data.get('date'):
                record_data['date'] = record_data['date'].date()
            
            record = MealEvent(**record_data)
            session.add(record)
            session.flush() # To get the ID for the vector DB
            
            # Prepare for vector embedding
            timestamp_obj = record_data.get('timestamp')
            date_str = timestamp_obj.strftime('%B %d, %Y') if timestamp_obj else 'an unknown date'
            
            text_chunk = f"On {date_str}, for {record_data.get('meal_type')}, the user ate: {record_data.get('food_items')}."
            metadata = {
                "source_table": "meal_events",
                "record_id": record.id,
                "timestamp": record_data['timestamp'].isoformat()
            }
            documents.append(text_chunk)
            metadatas.append(metadata)
            ids.append(f"meal_{record.id}")

    if documents:
        embeddings = embed_documents(documents)
        if embeddings:
            collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
            logger.info(f"Added {len(documents)} meal event embeddings to vector store.")

    session.commit()
    logger.info("Meal events ingestion complete.")

def ingest_activity_logs(session: Session, file_path: str):
    logger.info(f"Ingesting activity logs from {file_path}...")
    df = load_and_clean_csv(file_path, date_columns=['timestamp_start'])
    if df is None: return

    for _, row in df.iterrows():
        if row['timestamp_start'] is None:
            continue
        exists = session.query(ActivityLog).filter_by(timestamp_start=row['timestamp_start']).first()
        if not exists:
            record = ActivityLog(**row.to_dict())
            session.add(record)
    session.commit()
    logger.info("Activity logs ingestion complete.")

def ingest_sleep_logs(session: Session, file_path: str):
    logger.info(f"Ingesting sleep logs from {file_path}...")
    df = load_and_clean_csv(file_path, date_columns=['date', 'sleep_start', 'sleep_end'])
    if df is None: return

    for _, row in df.iterrows():
        if row['sleep_start'] is None:
            continue
        exists = session.query(SleepLog).filter_by(sleep_start=row['sleep_start']).first()
        if not exists:
            record_data = row.to_dict()
            if record_data.get('date'):
                record_data['date'] = record_data['date'].date()
            if record_data.get('was_disrupted') is not None:
                record_data['was_disrupted'] = bool(int(record_data['was_disrupted']))

            record = SleepLog(**record_data)
            session.add(record)
    session.commit()
    logger.info("Sleep logs ingestion complete.")

def ingest_cgm_aggregates(session: Session, collection, file_path: str):
    logger.info(f"Ingesting CGM aggregates from {file_path}...")
    df = load_and_clean_csv(file_path, date_columns=['date'])
    if df is None: return
    
    documents, metadatas, ids = [], [], []

    for _, row in df.iterrows():
        if row['date'] is None:
            continue
        # When querying, we must also use a date object for comparison
        exists = session.query(CGMAggregate).filter_by(date=row['date'].date()).first()
        if not exists:
            record_data = row.to_dict()
            # The date is guaranteed to be a valid datetime object here
            record_data['date'] = record_data['date'].date()

            record = CGMAggregate(**record_data)
            session.add(record)
            session.flush()

            # Prepare for vector embedding
            text_chunk = (
                f"On {record_data['date']}, daily glucose average was {record_data.get('mean_glucose', 'N/A'):.2f} mg/dL, "
                f"with a GMI of {record_data.get('gmi', 'N/A'):.2f}. Time in range was {record_data.get('time_in_range_pct', 'N/A'):.2f}%. "
                f"Flags: Hypo={bool(record_data.get('hypo_flag'))}, Hyper={bool(record_data.get('hyper_flag'))}."
            )
            metadata = {
                "source_table": "cgm_aggregates",
                "record_id": record.id,
                "timestamp": record_data['date'].isoformat()
            }
            documents.append(text_chunk)
            metadatas.append(metadata)
            ids.append(f"agg_{record.id}")

    if documents:
        embeddings = embed_documents(documents)
        if embeddings:
            collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
            logger.info(f"Added {len(documents)} CGM aggregate embeddings to vector store.")

    session.commit()
    logger.info("CGM aggregates ingestion complete.")

# --- MAIN EXECUTION ---
def main():
    logger.info("Starting data ingestion process...")
    db_session = get_db_session()
    chroma_collection = get_chroma_client()

    try:
        ingest_cgm_stream(db_session, os.path.join(CSV_DIR, 'cgm_stream.csv'))
        ingest_meal_events(db_session, chroma_collection, os.path.join(CSV_DIR, 'meal_events.csv'))
        ingest_activity_logs(db_session, os.path.join(CSV_DIR, 'activity_logs.csv'))
        ingest_sleep_logs(db_session, os.path.join(CSV_DIR, 'sleep_logs.csv'))
        ingest_cgm_aggregates(db_session, chroma_collection, os.path.join(CSV_DIR, 'cgm_aggregates.csv'))
        
        logger.info("All data ingested successfully.")
        
        # Verify vector store count
        count = chroma_collection.count()
        logger.info(f"Vector store now contains {count} documents.")

    except Exception as e:
        logger.error(f"An error occurred during ingestion: {e}", exc_info=True)
        db_session.rollback()
    finally:
        db_session.close()
        logger.info("Database session closed.")

if __name__ == "__main__":
    main() 