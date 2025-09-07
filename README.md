# PlantDoc 🌱

An AI-powered document Q&A chatbot using Retrieval-Augmented Generation (RAG) with specialized agricultural models. While originally designed for plant pathology, it can process any PDF document for domain-specific question answering.

## Overview

PlantDoc demonstrates how to build a specialized RAG system that combines semantic search with large language models. It processes PDF documents into searchable chunks and provides accurate, context-aware responses with source citations. The system uses agricultural-specialized embedding models but can adapt to any document domain.

## Features

- **Specialized Models**: Agricultural embedding model (`recobo/agri-sentence-transformer`) + Ollama Llama3.1:8b LLM
- **Document Processing**: Extracts and processes 948 pages into 2,870 searchable chunks
- **Vector Search**: ChromaDB-powered semantic search for relevant content retrieval
- **Conversation Memory**: Maintains context across follow-up questions
- **Source Citations**: Displays page numbers and relevant text excerpts
- **Web Interface**: Clean Streamlit-based user interface

## Quick Start

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- A PDF document to process (place in `data/raw/` directory)

### Installation

1. **Clone this repository**
   ```bash
  git clone https://github.com/fahmidhasann/PlantDoc-RAG-ChatBot
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pull the LLM model**
   ```bash
   ollama pull llama3.1:8b
   ```

4. **Initialize the database** (first time only)
   ```bash
   python setup.py
   ```

5. **Launch the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the interface** at `http://localhost:8501`

## Architecture

```
src/
├── document_processor.py    # PDF text extraction and chunking
├── embeddings.py           # Agricultural embedding model
├── llm.py                  # Ollama LLM interface  
├── vector_store.py         # ChromaDB vector operations
└── rag_engine.py          # RAG pipeline orchestration

data/
├── raw/                   # Source PDF document
├── processed/             # Chunked text cache
└── vector_db/            # ChromaDB persistence

config/
└── config.yaml           # Model and system configuration
```

## Configuration

Key settings in `config/config.yaml`:

```yaml
models:
  llm:
    name: "llama3.1:8b"
    base_url: "http://localhost:11434"
  embeddings:
    name: "recobo/agri-sentence-transformer"

document_processing:
  chunk_size: 1500
  chunk_overlap: 300

chat:
  temperature: 0.7
  max_tokens: 2048
  max_history: 10
```

## Usage Examples

Ask questions about your document content. For plant pathology documents:

- "What are the main types of plant pathogens?"
- "How do fungal diseases spread between plants?"
- "Explain the disease triangle concept"
- "What are the symptoms of bacterial wilt?"
- "How do environmental factors affect plant disease?"

The system adapts to any document domain - simply ask questions relevant to your PDF content.

## Technical Details

### RAG Pipeline
1. **Query Embedding**: Convert user question to vector representation
2. **Document Retrieval**: Semantic search across document chunks
3. **Context Assembly**: Combine relevant chunks with conversation history
4. **Response Generation**: LLM generates answer using retrieved context
5. **Source Attribution**: Return response with page references

### Models
- **LLM**: Llama3.1:8b (4.7GB) via Ollama for response generation
- **Embeddings**: recobo/agri-sentence-transformer (768-dim) specialized for agricultural content
- **Vector DB**: ChromaDB for efficient similarity search

### Performance
- **Setup Time**: ~10-15 minutes (includes model downloads)
- **Document Processing**: 2,870 chunks from 948 pages
- **Query Response**: 2-5 seconds average
- **Memory Usage**: ~4-6GB during operation

## Troubleshooting

### Common Issues

**"Model not found"**
- Ensure Ollama is running: `ollama serve`
- Pull the model: `ollama pull llama3.1:8b`

**"Vector database is empty"**
- Run setup: `python setup.py`
- Verify PDF location: `data/raw/document.pdf` (or update path in config)
- Ensure PDF is readable and not password-protected

**Slow performance**
- Minimum 8GB RAM recommended
- GPU acceleration improves embedding generation
- Check Ollama service status

### System Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB for models and database
- **Network**: Internet access for initial model downloads

## Development

### Adding New Documents
1. Place PDF in `data/raw/`
2. Update PDF path in `config/config.yaml`
3. Run `python setup.py` to reprocess

### Customizing Models
- Edit model names in `config/config.yaml`
- Restart application to load new models
- Rerun setup if changing embedding model

## Important Disclaimers

### Content Copyright
This software is designed to work with PDF documents for educational and research purposes. **Any PDF content you process remains subject to its original copyright and licensing terms.** Users must:

- Ensure they have proper permissions for any copyrighted materials
- Comply with fair use guidelines and institutional policies  
- Obtain necessary licenses for commercial or public use of processed content

### PDF Requirements
While this project was developed using a plant pathology textbook, **the software works with any PDF document**. To use your own content:

1. Place your PDF in `data/raw/` directory
2. Update the file path in `config/config.yaml`
3. Run `python setup.py` to process your document
4. The system will automatically adapt to your content domain

## License

**Software Code**: MIT License (see [LICENSE](LICENSE) file)

**Third-Party Content**: Not included - users must provide their own PDF documents and ensure proper licensing compliance.
