"""
Query rewriter module for conversational RAG.
This module handles query rewriting to maintain context across conversation turns.
"""
from typing import List, Dict, Optional
import logging
from .llm import LLMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryRewriter:
    def __init__(self, llm_model: LLMModel):
        self.llm_model = llm_model
        
        # System prompt for query rewriting
        self.rewriter_system_prompt = """You are a query rewriting assistant for a document search system. Your task is to rewrite user queries to be standalone and contextually complete based on the conversation history.

Instructions:
1. Read the conversation history to understand the context
2. If the current query is already complete and standalone, return it as is
3. If the current query is a follow-up question or lacks context, rewrite it to include the necessary context from the conversation history
4. The rewritten query should be clear, specific, and contain all necessary information for document retrieval
5. Focus on the main topic or subject being discussed in the conversation
6. Keep the rewritten query concise but comprehensive
7. Only output the rewritten query, nothing else

Examples:

Conversation History:
Human: What are the symptoms of late blight of potato?
Assistant: Late blight of potato is caused by Phytophthora infestans and shows symptoms like...

Current Query: What are the control measures?
Rewritten Query: What are the control measures for late blight of potato?

Conversation History:
Human: Tell me about bacterial canker in tomatoes
Assistant: Bacterial canker in tomatoes is caused by Clavibacter michiganensis...

Current Query: How do I treat it?
Rewritten Query: How do I treat bacterial canker in tomatoes?

Conversation History:
Human: What is plant pathology?
Assistant: Plant pathology is the scientific study of diseases in plants...

Current Query: What are the main types of diseases?
Rewritten Query: What are the main types of plant diseases?

Remember: Only return the rewritten query, nothing else."""

    def rewrite_query(
        self, 
        current_query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Rewrite a query to be contextually complete based on conversation history.
        
        Args:
            current_query: The user's current question
            conversation_history: List of previous conversation exchanges
            
        Returns:
            Rewritten query that includes necessary context
        """
        try:
            # If no conversation history or only one exchange, return original query
            if not conversation_history or len(conversation_history) < 1:
                logger.info("No conversation history, returning original query")
                return current_query
            
            # Format conversation history for the prompt
            history_text = self._format_history_for_rewriting(conversation_history)
            
            # Create the prompt for query rewriting
            prompt = f"""Conversation History:
{history_text}

Current Query: {current_query}
Rewritten Query:"""

            # Generate rewritten query
            rewritten_query = self.llm_model.generate_response(
                prompt=prompt,
                system_prompt=self.rewriter_system_prompt,
                temperature=0.3,  # Lower temperature for more consistent rewriting
                max_tokens=200    # Shorter responses for query rewriting
            )
            
            # Clean up the response (remove any extra text)
            rewritten_query = rewritten_query.strip()
            
            # If the rewriting failed or returned empty, use original query
            if not rewritten_query or len(rewritten_query) < 5:
                logger.warning("Query rewriting failed, using original query")
                return current_query
            
            logger.info(f"Query rewritten: '{current_query}' -> '{rewritten_query}'")
            return rewritten_query
            
        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            # Fallback to original query if rewriting fails
            return current_query
    
    def _format_history_for_rewriting(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for query rewriting context."""
        # Only use the last 2-3 exchanges to avoid context overload
        recent_history = history[-3:] if len(history) > 3 else history
        
        formatted_history: List[str] = []
        for exchange in recent_history:
            human_msg = exchange.get("human", "")
            ai_msg = exchange.get("ai", "")
            
            # Truncate AI responses to keep context focused
            if len(ai_msg) > 200:
                ai_msg = ai_msg[:200] + "..."
            
            formatted_history.append(f"Human: {human_msg}")
            formatted_history.append(f"Assistant: {ai_msg}")
        
        return "\n".join(formatted_history)
    
    def should_rewrite_query(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> bool:
        """
        Determine if a query needs rewriting based on simple heuristics.
        This can help optimize performance by avoiding unnecessary LLM calls.
        """
        if not conversation_history or len(conversation_history) < 1:
            return False
        
        query_lower = query.lower().strip()
        
        # Common follow-up patterns that likely need rewriting
        follow_up_indicators = [
            # Pronouns
            "what are the", "how do i", "can you", "what is the", "how does",
            "what about", "how about", "tell me about",
            # Relative references
            "this", "that", "these", "those", "it", "them", "they",
            # Treatment/solution questions without specific context
            "preventive measures", "treatment", "control", "management",
            "symptoms", "causes", "prevention",
            # Question words without specific subject
            "why", "how", "when", "where", "what"
        ]
        
        # Check if query starts with common follow-up patterns
        for indicator in follow_up_indicators:
            if query_lower.startswith(indicator):
                return True
        
        # Check if query is very short (likely incomplete)
        if len(query.split()) <= 4:
            return True
        
        return False
