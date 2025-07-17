import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
VECTOR_STORE_DIR = "vector_store"
CHROMA_COLLECTION = "buhai_rag_collection"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- EMBEDDING ---
class LocalEmbeddingFunction:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_query(self, query: str):
        return self.model.encode(query, convert_to_tensor=False).tolist()

if GEMINI_API_KEY:
    logger.info("Using Google Gemini embeddings for retriever.")
    # This should be replaced with a proper Gemini embedding class if available
    embedding_fn = LocalEmbeddingFunction() # Placeholder
else:
    logger.warning("GEMINI_API_KEY not found. Using dummy sentence transformer embeddings for retriever.")
    embedding_fn = LocalEmbeddingFunction()

class VectorRetriever:
    def __init__(self):
        """Initializes the retriever by connecting to the persistent vector store."""
        try:
            client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
            # We don't pass the embedding function at collection *get* time,
            # because the query method below will handle the embedding.
            self.collection = client.get_collection(name=CHROMA_COLLECTION)
            self.embedding_function = embedding_fn
            logger.info("Successfully connected to ChromaDB collection.")
            logger.info(f"Collection '{CHROMA_COLLECTION}' contains {self.collection.count()} items.")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}", exc_info=True)
            self.collection = None
            self.embedding_function = None

    def query(self, question: str, k: int = 5):
        """
        Queries the vector store to find the top k most relevant documents.

        Args:
            question (str): The user's question.
            k (int): The number of results to return.

        Returns:
            list: A list of dictionaries, where each dictionary contains
                  'text', 'metadata', and 'score'. Returns empty list on error.
        """
        if not self.collection:
            logger.error("Cannot perform query: collection is not available.")
            return []

        try:
            # First, we embed the query text using the selected embedding function.
            query_embedding = self.embedding_function.embed_query(question)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Reformat the results to be more user-friendly
            output = []
            if results and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    output.append({
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "score": results['distances'][0][i]
                    })
            return output
        except Exception as e:
            logger.error(f"An error occurred during query: {e}", exc_info=True)
            return []

# --- Example Usage ---
if __name__ == '__main__':
    # This assumes you have run ingest_csv.py first to populate the vector store
    retriever = VectorRetriever()
    
    if retriever.collection:
        # Example query
        test_question = "What did I eat for breakfast recently?"
        logger.info(f"Performing test query: '{test_question}'")
        
        retrieved_context = retriever.query(test_question, k=3)
        
        if retrieved_context:
            print("\n--- Query Results ---")
            for i, context in enumerate(retrieved_context):
                print(f"\nResult {i+1}:")
                print(f"  Text: {context['text']}")
                print(f"  Metadata: {context['metadata']}")
                print(f"  Score (distance): {context['score']:.4f}")
            print("\n---------------------\n")
        else:
            print("Query returned no results. Is the vector store populated?") 