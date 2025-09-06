"""
RAG engine that combines document retrieval with language model generation.
"""
from typing import List, Dict, Any, Optional, Tuple
import logging
from .embeddings import EmbeddingModel
from .llm import LLMModel
from .vector_store import VectorStore
from .query_rewriter import QueryRewriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(
        self, 
        embedding_model: EmbeddingModel,
        llm_model: LLMModel,
        vector_store: VectorStore,
        max_context_length: int = 4000,
        enable_query_rewriting: bool = True
    ):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.vector_store = vector_store
        self.max_context_length = max_context_length
        self.enable_query_rewriting = enable_query_rewriting
        
        # Initialize query rewriter if enabled
        if self.enable_query_rewriting:
            self.query_rewriter = QueryRewriter(llm_model)
        else:
            self.query_rewriter = None
        
        # System prompt for PlantDoc
        self.system_prompt = """You are PlantDoc, an AI assistant that helps users understand document content through question answering.

Instructions:
1. Answer questions based on the provided context from the document
2. If the context doesn't contain relevant information, say "I don't have enough information in the provided context to answer that question"
3. Always cite page numbers when available
4. Be precise and helpful in your responses
5. Focus on the content domain of the provided document
6. Use bullets and headings when necessary to make responses clear and organized

Context from the document:
{context}

Previous conversation history:
{history}

Please answer the following question based on the context and conversation history above."""
    
    def retrieve_relevant_documents(self, query: str, k: int = 15) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a given query."""
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_model.generate_embedding(query)
            
            # Search for similar documents
            results = self.vector_store.similarity_search(query_embedding, k=k)
            
            logger.info(f"Retrieved {len(results)} relevant documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise
    
    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string."""
        context_parts = []
        total_length = 0
        
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            page_num = metadata.get("page_number", "Unknown")
            
            formatted_doc = f"[Page {page_num}]\n{content}\n"
            
            # Check if adding this document would exceed max context length
            if total_length + len(formatted_doc) > self.max_context_length:
                break
                
            context_parts.append(formatted_doc)
            total_length += len(formatted_doc)
        
        return "\n---\n".join(context_parts)
    
    def format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for context."""
        if not history:
            return "No previous conversation."
        
        formatted_history = []
        for exchange in history[-3:]:  # Keep last 3 exchanges
            human_msg = exchange.get("human", "")
            ai_msg = exchange.get("ai", "")
            formatted_history.append(f"Human: {human_msg}")
            formatted_history.append(f"Assistant: {ai_msg}")
        
        return "\n".join(formatted_history)
    
    def generate_response(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate response using RAG approach with optional query rewriting."""
        try:
            # Rewrite query for better context if enabled
            search_query = query
            if self.enable_query_rewriting and self.query_rewriter:
                search_query = self.query_rewriter.rewrite_query(query, conversation_history)
                logger.info(f"Using search query: '{search_query}' (original: '{query}')")
            
            # Retrieve relevant documents using the (potentially rewritten) query
            relevant_docs = self.retrieve_relevant_documents(search_query)
            
            # Format context and history
            context = self.format_context(relevant_docs)
            history = self.format_conversation_history(conversation_history or [])
            
            # Create the full prompt (use original query for response generation)
            system_prompt = self.system_prompt.format(
                context=context,
                history=history
            )
            
            # Generate response using the original query
            response = self.llm_model.generate_response(
                prompt=query,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            logger.info("Generated RAG response successfully")
            return response, relevant_docs
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            raise
    
    def generate_streaming_response(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """Generate streaming response using RAG approach with optional query rewriting."""
        try:
            # Rewrite query for better context if enabled
            search_query = query
            if self.enable_query_rewriting and self.query_rewriter:
                search_query = self.query_rewriter.rewrite_query(query, conversation_history)
                logger.info(f"Using search query: '{search_query}' (original: '{query}')")
            
            # Retrieve relevant documents using the (potentially rewritten) query
            relevant_docs = self.retrieve_relevant_documents(search_query)
            
            # Format context and history
            context = self.format_context(relevant_docs)
            history = self.format_conversation_history(conversation_history or [])
            
            # Create the full prompt (use original query for response generation)
            system_prompt = self.system_prompt.format(
                context=context,
                history=history
            )
            
            # Generate streaming response using the original query
            for chunk in self.llm_model.generate_streaming_response(
                prompt=query,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            ):
                yield chunk
            
            logger.info("Generated streaming RAG response successfully")
            
        except Exception as e:
            logger.error(f"Error generating streaming RAG response: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        return {
            "vector_store_stats": self.vector_store.get_collection_stats(),
            "embedding_dimension": self.embedding_model.get_embedding_dimension(),
            "llm_model": self.llm_model.model_name,
            "embedding_model": self.embedding_model.model_name
        }
