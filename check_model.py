#!/usr/bin/env python3
"""
Script to check which model is currently configured and loaded.
"""
import yaml
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_config_model():
    """Check which model is configured in config.yaml"""
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        print("=== Configuration Check ===")
        print(f"🧠 LLM Model: {config['models']['llm']['name']}")
        print(f"🔗 Base URL: {config['models']['llm']['base_url']}")
        print(f"🤖 Embedding Model: {config['models']['embeddings']['name']}")
        return config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

def check_model_availability():
    """Check if the configured model is available in Ollama"""
    try:
        from src.llm import LLMModel
        
        # Load config
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        print("\n=== Model Availability Check ===")
        print(f"📡 Testing connection to {config['models']['llm']['name']}...")
        
        # Initialize LLM model (this will test the connection)
        llm_model = LLMModel(
            model_name=config['models']['llm']['name'],
            base_url=config['models']['llm']['base_url']
        )
        
        print(f"✅ Successfully connected to {llm_model.model_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to model: {e}")
        return False

def check_rag_system():
    """Check the RAG system stats to see which model is being used"""
    try:
        from src.document_processor import DocumentProcessor
        from src.embeddings import EmbeddingModel
        from src.llm import LLMModel
        from src.vector_store import VectorStore
        from src.rag_engine import RAGEngine
        
        # Load config
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        print("\n=== RAG System Check ===")
        
        # Initialize models
        embedding_model = EmbeddingModel(
            model_name=config["models"]["embeddings"]["name"]
        )
        
        llm_model = LLMModel(
            model_name=config["models"]["llm"]["name"],
            base_url=config["models"]["llm"]["base_url"]
        )
        
        # Initialize vector store
        vector_store = VectorStore(
            persist_directory=config["vector_db"]["persist_directory"],
            collection_name=config["vector_db"]["collection_name"]
        )
        
        # Initialize RAG engine
        rag_engine = RAGEngine(
            embedding_model=embedding_model,
            llm_model=llm_model,
            vector_store=vector_store,
            enable_query_rewriting=config["chat"].get("enable_query_rewriting", True)
        )
        
        # Get stats
        stats = rag_engine.get_stats()
        
        print("📊 System Statistics:")
        print(f"  🧠 LLM Model: {stats.get('llm_model', 'Unknown')}")
        print(f"  🤖 Embedding Model: {stats.get('embedding_model', 'Unknown')}")
        print(f"  📏 Embedding Dimension: {stats.get('embedding_dimension', 'Unknown')}")
        
        vector_stats = stats.get('vector_store_stats', {})
        print(f"  📚 Total Documents: {vector_stats.get('total_documents', 0)}")
        print(f"  🗂️ Collection: {vector_stats.get('collection_name', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking RAG system: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Checking Model Configuration...\n")
    
    # Check config
    config = check_config_model()
    if not config:
        sys.exit(1)
    
    # Check model availability
    model_available = check_model_availability()
    
    # Check RAG system (if model is available)
    if model_available:
        rag_working = check_rag_system()
        if rag_working:
            print("\n✅ All checks passed! Your model is ready to use.")
        else:
            print("\n⚠️ Model is available but RAG system has issues.")
    else:
        print(f"\n⚠️ Model '{config['models']['llm']['name']}' is not available.")
        print("💡 Try running: ollama pull gemma3:4b")
