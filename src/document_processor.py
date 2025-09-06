"""
Document processing module for PDF text extraction and chunking.
"""
import os
import pickle
from typing import List, Dict, Any, Optional
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            text += f"\n\n--- Page {page_num + 1} ---\n\n"
                            text += page_text
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                        continue
                
                logger.info(f"Successfully extracted text from {len(pdf_reader.pages)} pages")
                return text
                
        except Exception as e:
            logger.error(f"Error reading PDF file: {e}")
            raise
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        chunks = self.text_splitter.split_text(text)
        
        # Add metadata to each chunk
        chunked_documents = []
        for i, chunk in enumerate(chunks):
            # Skip chunks that are mostly page headers
            if self._is_page_header_only(chunk):
                continue
                
            # Extract page number from chunk if available
            page_num = self._extract_page_number(chunk)
            
            chunk_doc = {
                "content": chunk,
                "metadata": {
                    "chunk_id": i,
                    "page_number": page_num,
                    "source": "Plant Pathology Fifth Edition",
                    "chunk_size": len(chunk)
                }
            }
            chunked_documents.append(chunk_doc)
        
        logger.info(f"Created {len(chunked_documents)} text chunks")
        return chunked_documents
    
    def _is_page_header_only(self, text: str) -> bool:
        """Check if chunk contains only page headers and minimal content."""
        # Remove whitespace and newlines for analysis
        clean_text = text.strip()
        
        # Skip if too short
        if len(clean_text) < 50:
            return True
        
        # Skip if it's just page markers
        lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
        content_lines = [line for line in lines if not line.startswith('--- Page')]
        
        # If less than 30% is actual content, skip it
        if len(content_lines) < len(lines) * 0.3:
            return True
            
        # Check content quality - if mostly just page numbers and headers
        content_text = ' '.join(content_lines)
        if len(content_text) < 30:
            return True
            
        return False
    
    def _extract_page_number(self, text: str) -> int:
        """Extract page number from chunk text."""
        lines = text.split('\n')
        for line in lines:
            if line.strip().startswith('--- Page '):
                try:
                    page_num = int(line.split('Page ')[1].split(' ---')[0])
                    return page_num
                except:
                    continue
        return 1  # Default to page 1 if not found
    
    def save_processed_data(self, chunks: List[Dict[str, Any]], output_path: str):
        """Save processed chunks to disk."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(chunks, f)
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")
    
    def load_processed_data(self, input_path: str) -> List[Dict[str, Any]]:
        """Load processed chunks from disk."""
        with open(input_path, 'rb') as f:
            chunks = pickle.load(f)
        logger.info(f"Loaded {len(chunks)} chunks from {input_path}")
        return chunks
    
    def process_document(self, pdf_path: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Complete document processing pipeline."""
        logger.info(f"Processing document: {pdf_path}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        # Create chunks
        chunks = self.chunk_text(text)
        
        # Save if output path provided
        if output_path:
            self.save_processed_data(chunks, output_path)
        
        return chunks
