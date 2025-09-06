"""
Setup script to process the PDF and initialize the vector database.
"""
import os
import sys
import yaml
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main setup function."""
    try:
        # Load configuration
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        logger.info("Starting database setup...")
        
        # Initialize document processor
        doc_processor = DocumentProcessor(
            chunk_size=config["document_processing"]["chunk_size"],
            chunk_overlap=config["document_processing"]["chunk_overlap"]
        )
        
        # Process PDF
        pdf_path = config["paths"]["pdf_file"]
        processed_path = os.path.join(config["paths"]["processed_data"], "chunks.pkl")
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return False
        
        logger.info(f"Processing PDF: {pdf_path}")
        chunks = doc_processor.process_document(pdf_path, processed_path)
        
        # Initialize embedding model
        logger.info("Initializing embedding model...")
        embedding_model = EmbeddingModel(
            model_name=config["models"]["embeddings"]["name"]
        )
        
        # Generate embeddings
        logger.info("Generating embeddings for text chunks...")
        texts = [chunk["content"] for chunk in chunks]
        embeddings = embedding_model.generate_embeddings_batch(texts)
        
        # Initialize vector store and add documents
        logger.info("Storing embeddings in vector database...")
        vector_store = VectorStore(
            persist_directory=config["vector_db"]["persist_directory"],
            collection_name=config["vector_db"]["collection_name"]
        )
        
        vector_store.add_documents(chunks, embeddings)
        
        # Get final stats
        stats = vector_store.get_collection_stats()
        logger.info(f"Setup complete! Database contains {stats['total_documents']} documents")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during setup: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Database setup completed successfully!")
        print("You can now run the Streamlit app with: streamlit run app.py")
    else:
        print("\n❌ Database setup failed!")
        sys.exit(1)
