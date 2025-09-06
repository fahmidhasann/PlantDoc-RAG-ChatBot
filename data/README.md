# Data Directory

This directory contains the documents and processed data for your RAG system.

## Structure

```
data/
├── raw/          # Place your PDF documents here
├── processed/    # Automatically generated chunked text cache
└── vector_db/    # ChromaDB vector database storage
```

## Setup Instructions

1. **Add Your PDF**: Place your PDF document in the `raw/` folder
2. **Update Config**: Edit `config/config.yaml` to point to your PDF filename
3. **Run Setup**: Execute `python setup.py` to process your document

## Supported Documents

- Any PDF document (textbooks, research papers, manuals, etc.)
- Ensure PDFs are text-based (not scanned images)
- Remove password protection before processing

## Copyright Notice

**Important**: You are responsible for ensuring you have proper permissions to process any copyrighted documents. This software does not include or distribute any copyrighted content.
