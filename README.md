# College Articulation PDF Vector Database API

A FastAPI service for searching college articulation PDFs using Pinecone vector database. Returns raw search results designed to be used as a tool by Gemini or other LLMs.

## Features

- 📄 Automatic PDF text extraction and chunking
- 🔍 Semantic search using Pinecone with automatic embeddings (llama-text-embed-v2)
- 🚀 FastAPI REST endpoints
- 📊 Source citations and similarity scores for each chunk
- 🧩 Tool-ready: Returns structured data perfect for LLM integration

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set environment variables:**
Create a `.env` file with:
```
PINECONE_API_KEY=your_pinecone_api_key_here
```

3. **Run the API:**
```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### `GET /`
Health check endpoint.

### `GET /health`
Detailed health check with index statistics.

### `POST /ingest`
Ingest PDF(s) into Pinecone vector database.

**Request body (optional):**
```json
{
  "pdf_path": "pdfs/AA_UF.pdf"  // Optional: specific PDF path. If omitted, ingests all PDFs in pdfs/ folder
}
```

**Response:**
```json
{
  "message": "Successfully ingested 1 PDF(s) with 150 total chunks",
  "ingested_count": 150,
  "pdfs_processed": ["AA_UF"]
}
```

### `POST /search` or `POST /query`
Search the college PDF vector database and return relevant chunks.

**Request body:**
```json
{
  "query": "What are the articulation agreements for University of Florida?",
  "top_k": 5  // Optional: number of chunks to retrieve (default: 5)
}
```

**Response:**
```json
{
  "query": "What are the articulation agreements for University of Florida?",
  "chunks": [
    {
      "text": "The University of Florida has articulation agreements...",
      "source": "AA_UF",
      "score": 0.92,
      "id": "AA_UF_0"
    },
    {
      "text": "Transfer credits from community colleges...",
      "source": "AA_UF",
      "score": 0.88,
      "id": "AA_UF_1"
    }
  ],
  "total_results": 2,
  "context": "[Source: AA_UF]\nThe University of Florida has articulation agreements...\n\n---\n\n[Source: AA_UF]\nTransfer credits from community colleges..."
}
```

### `GET /stats`
Get statistics about the Pinecone index.

## Usage Example

1. **Ingest all PDFs:**
```bash
curl -X POST http://localhost:8000/ingest
```

2. **Search the vector database:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the articulation agreements for Florida universities?", "top_k": 5}'
```

## Using as a Gemini Tool

This API is designed to be used as a tool by Gemini. The response format provides:
- **`chunks`**: Array of relevant document chunks with metadata
- **`context`**: Pre-formatted combined context string ready for LLM consumption
- **`sources`**: Source PDF names for citation
- **`scores`**: Similarity scores for relevance ranking

Example Gemini function calling:
```python
# Gemini can call this API and use the returned chunks/context
# to generate answers based on the retrieved documents
```

## Architecture

- **Pinecone**: Vector database with automatic embeddings using `llama-text-embed-v2`
- **PyPDF2**: PDF text extraction
- **FastAPI**: REST API framework

## Notes

- The index is automatically created if it doesn't exist
- PDFs are chunked with 1000 character chunks and 200 character overlap
- Each chunk includes metadata with the source PDF name
- The API automatically handles embedding generation through Pinecone's integrated model
- Returns raw search results - no LLM integration (designed to be used as a tool)

# articulation-vector-db-api
