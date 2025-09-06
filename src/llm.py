"""
LLM module for generating responses using local Ollama models.
"""
import ollama  # type: ignore
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMModel:
    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama and model availability."""
        try:
            # Test if we can connect to Ollama
            models = self.client.list()
            available_models = [model['name'] for model in models['models']]
            
            if self.model_name not in available_models:
                logger.warning(f"Model {self.model_name} not found. Available models: {available_models}")
                logger.info(f"You may need to pull the model: ollama pull {self.model_name}")
            else:
                logger.info(f"Successfully connected to {self.model_name}")
                
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
            raise
    
    def generate_response(
        self, 
        prompt: str, 
        temperature: float = 0.7, 
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response from the LLM."""
        try:
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user", 
                "content": prompt
            })
            
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            
            return response['message']['content']  # type: ignore
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def generate_streaming_response(
        self, 
        prompt: str, 
        temperature: float = 0.7, 
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None
    ):
        """Generate streaming response from the LLM."""
        try:
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user", 
                "content": prompt
            })
            
            stream = self.client.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
            
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            raise
