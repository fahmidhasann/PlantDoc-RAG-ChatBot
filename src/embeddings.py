"""
Embeddings module for generating vector representations using Hugging Face sentence-transformers.
"""
from sentence_transformers import SentenceTransformer
from typing import List, Any
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, model_name: str = "recobo/agri-sentence-transformer"):
        self.model_name = model_name
        # Check if CUDA is available for GPU acceleration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load the model
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            logger.info("Falling back to a default model...")
            # Fallback to a smaller model if the agricultural one fails
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            logger.info("Loaded fallback model: all-MiniLM-L6-v2")
    
    def _test_connection(self):
        """Test model functionality."""
        try:
            # Test embedding generation with a simple text
            test_embedding = self.model.encode("test agricultural text", convert_to_tensor=False)
            logger.info(f"Model test successful. Embedding dimension: {len(test_embedding)}")
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            # Generate embedding using sentence-transformers
            embedding = self.model.encode(text, convert_to_tensor=False, convert_to_numpy=True)
            # Convert numpy array to list of floats
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # Use batch encoding for efficiency
            embeddings = self.model.encode(
                texts, 
                convert_to_tensor=False, 
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=32
            )
            
            # Convert numpy array to list of lists
            embeddings_list = [embedding.tolist() for embedding in embeddings]
            
            logger.info(f"Generated {len(embeddings_list)} embeddings")
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            # Fallback to individual processing
            logger.info("Falling back to individual embedding generation")
            embeddings = []
            total = len(texts)
            
            for i, text in enumerate(texts):
                if i % 10 == 0:  # Log progress every 10 items
                    logger.info(f"Processing embedding {i+1}/{total}")
                
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from this model."""
        test_embedding = self.generate_embedding("test")
        return len(test_embedding)
