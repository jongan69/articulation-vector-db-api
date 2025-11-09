import os
import glob
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Setup ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is required")

pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "college-pdf-knowledge"

# Create index if missing
# Using standard index creation (not create_index_for_model)
# Dimension 768 is standard for many embedding models
try:
    # Try v5.x API first (list_indexes returns Index objects)
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        try:
            pc.create_index(
                name=INDEX_NAME,
                dimension=768,  # Standard dimension for many embedding models
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )
        except Exception as e:
            print(f"Warning: Could not create index {INDEX_NAME}: {e}")
            print("Index may already exist or there may be a configuration issue.")
except (AttributeError, TypeError) as e:
    # Fallback for older API versions
    try:
        if hasattr(pc, 'has_index') and not pc.has_index(INDEX_NAME):
            pc.create_index(
                name=INDEX_NAME,
                dimension=768,
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )
    except Exception as e:
        print(f"Warning: Could not create index {INDEX_NAME}: {e}")
        print("Index may already exist or will be created manually.")

# Get index - this will fail if index doesn't exist, but that's okay
# The app will still start and users can create the index manually
try:
    index = pc.Index(INDEX_NAME)
except Exception as e:
    print(f"Warning: Could not connect to index {INDEX_NAME}: {e}")
    print("The index may need to be created manually in Pinecone dashboard.")
    index = None

# --- FastAPI App ---
app = FastAPI(
    title="College PDF Vector Database API",
    description="Vector database API for college articulation PDFs using Pinecone. Returns raw search results for use as a tool.",
    version="1.0.0"
)

# --- Pydantic Models ---
class ChunkResult(BaseModel):
    text: str
    source: str
    score: float
    id: str

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    query: str
    chunks: List[ChunkResult]
    total_results: int
    context: str  # Combined context from all chunks for easy use

class IngestRequest(BaseModel):
    pdf_path: Optional[str] = None  # If None, ingests all PDFs in pdfs/ folder

class IngestResponse(BaseModel):
    message: str
    ingested_count: int
    pdfs_processed: List[str]

# --- Helper: extract text from PDF ---
def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF {path}: {str(e)}")

# --- Helper: chunk text ---
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# --- Helper: generate embeddings ---
def generate_embedding(text: str) -> List[float]:
    """Generate embedding using hash-based method (simple and reliable)."""
    import hashlib
    
    # Create a deterministic embedding from text using hash
    # This is a simple approach - for production, consider using sentence-transformers or OpenAI embeddings
    hash_obj = hashlib.sha256(text.encode('utf-8'))
    hash_bytes = hash_obj.digest()
    
    # Create 768-dim vector by repeating hash pattern
    embedding = []
    for i in range(768):
        # Use modulo to cycle through hash bytes
        byte_idx = i % len(hash_bytes)
        # Normalize to [-1, 1] range
        val = (hash_bytes[byte_idx] / 255.0) * 2.0 - 1.0
        embedding.append(val)
    
    return embedding

# --- Step 1: Ingest PDF into Pinecone ---
def ingest_pdf(pdf_path: str, pdf_title: str) -> int:
    """Ingest a single PDF into Pinecone."""
    if index is None:
        raise HTTPException(status_code=500, detail="Pinecone index is not available. Please ensure the index exists.")
    
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    
    items = []
    for i, chunk in enumerate(chunks):
        # Generate embedding for each chunk
        embedding = generate_embedding(chunk)
        items.append({
            "id": f"{pdf_title}_{i}",
            "values": embedding,  # Vector values
            "metadata": {
                "source": pdf_title,
                "pdf_path": pdf_path,
                "text": chunk  # Store text in metadata for retrieval
            }
        })
    
    index.upsert(items)
    return len(chunks)

# --- Step 2: Query Pinecone ---
def search_vector_db(query: str, top_k: int = 5) -> List[ChunkResult]:
    """Query Pinecone vector database and return raw results."""
    if index is None:
        raise HTTPException(status_code=500, detail="Pinecone index is not available. Please ensure the index exists.")
    
    # Generate embedding for query
    query_embedding = generate_embedding(query)
    
    results = index.query(
        top_k=top_k,
        vector=query_embedding,  # Use vector instead of text
        include_values=False,
        include_metadata=True,
    )
    
    chunks = []
    
    for match in results["matches"]:
        # Get text from metadata (most reliable)
        chunk_text = ""
        source = "Unknown"
        chunk_id = match.get("id", "")
        score = match.get("score", 0.0)
        
        if "metadata" in match:
            chunk_text = match["metadata"].get("text", "")
            source = match["metadata"].get("source", "Unknown")
        
        # Fallback: try to get from match directly
        if not chunk_text:
            chunk_text = match.get("chunk_text", "")
        
        if chunk_text:
            chunks.append(ChunkResult(
                text=chunk_text,
                source=source,
                score=score,
                id=chunk_id
            ))
    
    return chunks

# --- API Endpoints ---
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "College PDF Vector Database API",
        "index": INDEX_NAME
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        if index is None:
            return {
                "status": "degraded",
                "index": INDEX_NAME,
                "error": "Index not available",
                "message": "Index may not exist or be accessible. Use /ingest to create and populate the index."
            }
        # Check if index is accessible
        stats = index.describe_index_stats()
        return {
            "status": "healthy",
            "index": INDEX_NAME,
            "index_stats": stats
        }
    except Exception as e:
        # Return partial health status instead of failing completely
        return {
            "status": "degraded",
            "index": INDEX_NAME,
            "error": str(e),
            "message": "Index may not exist or be accessible. Use /ingest to create and populate the index."
        }

@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdfs(request: Optional[IngestRequest] = None):
    """Ingest PDF(s) into Pinecone vector database."""
    pdfs_dir = "pdfs"
    
    if request and request.pdf_path:
        # Ingest a specific PDF
        if not os.path.exists(request.pdf_path):
            raise HTTPException(status_code=404, detail=f"PDF not found: {request.pdf_path}")
        
        pdf_files = [request.pdf_path]
    else:
        # Ingest all PDFs in pdfs/ folder
        pdf_files = glob.glob(f"{pdfs_dir}/*.pdf")
        if not pdf_files:
            # Check if directory exists
            if not os.path.exists(pdfs_dir):
                raise HTTPException(
                    status_code=404, 
                    detail=f"PDFs directory '{pdfs_dir}/' not found on server. PDFs need to be uploaded to the server or accessed from cloud storage."
                )
            raise HTTPException(
                status_code=404, 
                detail=f"No PDF files found in {pdfs_dir}/. Make sure PDFs are available on the server."
            )
    
    total_chunks = 0
    processed_pdfs = []
    errors = []
    
    for pdf_file in pdf_files:
        try:
            pdf_title = os.path.basename(pdf_file).replace(".pdf", "")
            chunks_count = ingest_pdf(pdf_file, pdf_title)
            total_chunks += chunks_count
            processed_pdfs.append(pdf_title)
        except Exception as e:
            # Continue with other PDFs even if one fails
            error_msg = f"Error ingesting {pdf_file}: {str(e)}"
            errors.append(error_msg)
            print(error_msg)
            continue
    
    if not processed_pdfs:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest any PDFs. Errors: {'; '.join(errors) if errors else 'Unknown error'}"
        )
    
    message = f"Successfully ingested {len(processed_pdfs)} PDF(s) with {total_chunks} total chunks"
    if errors:
        message += f". {len(errors)} PDF(s) failed to ingest."
    
    return IngestResponse(
        message=message,
        ingested_count=total_chunks,
        pdfs_processed=processed_pdfs
    )

@app.post("/search", response_model=QueryResponse)
async def search_college_data(request: QueryRequest):
    """Search the college PDF vector database and return relevant chunks."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        chunks = search_vector_db(request.query, top_k=request.top_k)
        
        # Combine all chunks into a single context string
        context = "\n\n---\n\n".join([f"[Source: {chunk.source}]\n{chunk.text}" for chunk in chunks])
        
        return QueryResponse(
            query=request.query,
            chunks=chunks,
            total_results=len(chunks),
            context=context
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_college_data(request: QueryRequest):
    """Alias for /search endpoint for backward compatibility."""
    return await search_college_data(request)

@app.get("/stats")
async def get_stats():
    """Get statistics about the Pinecone index."""
    try:
        if index is None:
            return {
                "index_name": INDEX_NAME,
                "error": "Index not available",
                "message": "Index may not exist or be accessible. Use /ingest to create and populate the index."
            }
        stats = index.describe_index_stats()
        return {
            "index_name": INDEX_NAME,
            "stats": stats
        }
    except Exception as e:
        return {
            "index_name": INDEX_NAME,
            "error": str(e),
            "message": "Index may not exist or be accessible. Use /ingest to create and populate the index."
        }

# --- Example usage ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

