"""
Main Streamlit application for PlantDoc - the Plant Pathology RAG Chatbot.
"""
import streamlit as st
import yaml
import os
import sys
from typing import List, Dict, Any
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingModel
from src.llm import LLMModel
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="PlantDoc",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_config():
    """Load configuration from YAML file."""
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

@st.cache_resource
def initialize_rag_system(config):
    """Initialize the RAG system components."""
    try:
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
        
        # Check if vector store is empty and needs to be populated
        stats = vector_store.get_collection_stats()
        if stats["total_documents"] == 0:
            st.warning("Vector database is empty. Please run the setup to process the PDF first.")
            return None
        
        # Initialize RAG engine
        rag_engine = RAGEngine(
            embedding_model=embedding_model,
            llm_model=llm_model,
            vector_store=vector_store,
            enable_query_rewriting=config["chat"].get("enable_query_rewriting", True)
        )
        
        return rag_engine
        
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None

def setup_database(config):
    """Setup the vector database by processing the PDF."""
    try:
        with st.spinner("Processing PDF and creating embeddings..."):
            # Initialize document processor
            doc_processor = DocumentProcessor(
                chunk_size=config["document_processing"]["chunk_size"],
                chunk_overlap=config["document_processing"]["chunk_overlap"]
            )
            
            # Process PDF
            pdf_path = config["paths"]["pdf_file"]
            processed_path = os.path.join(config["paths"]["processed_data"], "chunks.pkl")
            
            if not os.path.exists(pdf_path):
                st.error(f"PDF file not found: {pdf_path}")
                return False
            
            chunks = doc_processor.process_document(pdf_path, processed_path)
            
            # Initialize embedding model
            embedding_model = EmbeddingModel(
                model_name=config["models"]["embeddings"]["name"]
            )
            
            # Generate embeddings
            texts = [chunk["content"] for chunk in chunks]
            embeddings = embedding_model.generate_embeddings_batch(texts)
            
            # Initialize vector store and add documents
            vector_store = VectorStore(
                persist_directory=config["vector_db"]["persist_directory"],
                collection_name=config["vector_db"]["collection_name"]
            )
            
            vector_store.add_documents(chunks, embeddings)
            
            st.success(f"Successfully processed {len(chunks)} chunks and stored embeddings!")
            return True
            
    except Exception as e:
        st.error(f"Error setting up database: {e}")
        return False

def main():
    """Main application function."""
    # Load configuration
    config = load_config()
    
    # Initialize session state first
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_count" not in st.session_state:
        st.session_state.conversation_count = 1
    
    # App header
    st.title("🌱 PlantDoc")
    st.markdown("### Your AI-Powered Document Assistant")
    st.markdown("Ask questions about your PDF documents and get intelligent answers with source citations! 📚✨")
    
    # Add some spacing
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📚 About")
        st.markdown("""
        **PlantDoc** is your AI-powered document assistant! 🤖
        
        📖 **What it does:**
        - Answers questions about your PDF documents
        - Uses advanced AI to find relevant information
        - Provides source citations for transparency
        
        🚀 **Getting Started:**
        1. Click "Setup Database" if this is your first time
        2. Wait for the system to process your document
        3. Start asking questions!
        """)
        
        st.header("🔧 System Status")
        
        # Check if database needs setup
        st.markdown("**Database Setup:**")
        if st.button("🚀 Setup Database", help="Process your PDF and create the knowledge base", type="primary"):
            if setup_database(config):
                st.rerun()
        
        st.caption("💡 First time here? Click 'Setup Database' to get started!")
        
        # Initialize RAG system
        rag_engine = initialize_rag_system(config)
        
        if rag_engine:
            stats = rag_engine.get_stats()
            st.success("✅ System Ready")
            
            # Display user-friendly system information
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="📄 Documents",
                    value=f"{stats.get('vector_store_stats', {}).get('total_documents', 0):,}",
                    help="Number of text chunks available for search"
                )
            
            with col2:
                st.metric(
                    label="🧠 AI Model",
                    value=stats.get('llm_model', 'Unknown'),
                    help="Language model being used"
                )
            
            # Additional info in a clean format
            with st.expander("🔧 Technical Details", expanded=False):
                vector_stats = stats.get('vector_store_stats', {})
                st.write(f"**📊 Collection:** {vector_stats.get('collection_name', 'N/A')}")
                st.write(f"**📏 Embedding Dimension:** {stats.get('embedding_dimension', 'N/A')}")
                st.write(f"**🤖 Embedding Model:** {stats.get('embedding_model', 'N/A')}")
                st.write(f"**💾 Database Location:** `{vector_stats.get('persist_directory', 'N/A')}`")
        else:
            st.error("❌ System Not Ready")
            st.stop()
        
        st.header("💬 Conversation")
        
        # Show current conversation number
        st.caption(f"📋 Conversation #{st.session_state.conversation_count}")
        
        # New conversation button
        if st.button("🆕 New Conversation", help="Start a fresh conversation"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.session_state.conversation_count += 1
            st.success(f"Started conversation #{st.session_state.conversation_count}!")
            st.rerun()
        
        # Clear current conversation button
        if st.button("🗑️ Clear Current Chat", help="Clear the current conversation"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.success("Current conversation cleared!")
            st.rerun()
        
        # Show conversation stats
        if "messages" in st.session_state and st.session_state.messages:
            st.caption(f"💭 Messages in chat: {len(st.session_state.messages)}")
            if "conversation_history" in st.session_state:
                st.caption(f"🔄 Exchanges: {len(st.session_state.conversation_history)}")
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages in a user-friendly way
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander(f"📖 Sources ({len(message['sources'])} found)", expanded=False):
                    for i, source in enumerate(message["sources"]):
                        page_num = source['metadata'].get('page_number', 'Unknown')
                        st.markdown(f"""
                        **📄 Source {i+1}** (Page {page_num})
                        > {source['content'][:300]}{"..." if len(source['content']) > 300 else ""}
                        """)
    
    # Chat input
    if not st.session_state.messages:
        st.info("👋 Welcome! Ask me anything about your document. Try questions like:\n"
                "- What is this document about?\n"
                "- Can you summarize the main points?\n"
                "- Tell me about [specific topic]")
    
    if prompt := st.chat_input("💬 Ask me anything about your document..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking and searching through your document..."):
                try:
                    # Show debug info if query rewriting is enabled
                    if config["chat"].get("enable_query_rewriting", True) and st.session_state.conversation_history:
                        with st.expander("🔍 Query Processing", expanded=False):
                            st.caption("💡 Processing your question with conversation context...")
                    
                    response, sources = rag_engine.generate_response(
                        query=prompt,
                        conversation_history=st.session_state.conversation_history,
                        temperature=config["chat"]["temperature"],
                        max_tokens=config["chat"]["max_tokens"]
                    )
                    
                    st.markdown(response)
                    
                    # Show sources in a more user-friendly way
                    if sources:
                        with st.expander(f"📖 Sources ({len(sources)} found)", expanded=False):
                            for i, source in enumerate(sources):
                                page_num = source['metadata'].get('page_number', 'Unknown')
                                st.markdown(f"""
                                **📄 Source {i+1}** (Page {page_num})
                                > {source['content'][:300]}{"..." if len(source['content']) > 300 else ""}
                                """)
                    
                    # Add to session state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": sources
                    })
                    
                    # Update conversation history
                    st.session_state.conversation_history.append({
                        "human": prompt,
                        "ai": response
                    })
                    
                    # Keep only recent history
                    if len(st.session_state.conversation_history) > config["chat"]["max_history"]:
                        st.session_state.conversation_history = st.session_state.conversation_history[-config["chat"]["max_history"]:]
                
                except Exception as e:
                    st.error(f"❌ Oops! Something went wrong: {e}")
                    st.info("💡 Try rephrasing your question or check if the system is properly set up.")

if __name__ == "__main__":
    main()
