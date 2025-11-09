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
try:
    # Try v5.x API first (list_indexes returns Index objects)
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        pc.create_index_for_model(
            name=INDEX_NAME,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "chunk_text"}
            },
        )
except (AttributeError, TypeError):
    # Fallback for older API versions that use has_index()
    try:
        if not pc.has_index(INDEX_NAME):
            pc.create_index_for_model(
                name=INDEX_NAME,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": "llama-text-embed-v2",
                    "field_map": {"text": "chunk_text"}
                },
            )
    except AttributeError:
        # If neither method exists, assume index exists or will be created manually
        pass

index = pc.Index(INDEX_NAME)

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

# --- Step 1: Ingest PDF into Pinecone ---
def ingest_pdf(pdf_path: str, pdf_title: str) -> int:
    """Ingest a single PDF into Pinecone."""
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    
    items = [
        {
            "id": f"{pdf_title}_{i}",
            "chunk_text": chunk,
            "metadata": {
                "source": pdf_title,
                "pdf_path": pdf_path,
                "text": chunk  # Store text in metadata for reliable retrieval
            }
        }
        for i, chunk in enumerate(chunks)
    ]
    
    index.upsert(items)
    return len(items)

# --- Step 2: Query Pinecone ---
def search_vector_db(query: str, top_k: int = 5) -> List[ChunkResult]:
    """Query Pinecone vector database and return raw results."""
    results = index.query(
        top_k=top_k,
        text=query,  # Pinecone automatically embeds this text
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
        # Check if index is accessible
        stats = index.describe_index_stats()
        return {
            "status": "healthy",
            "index": INDEX_NAME,
            "index_stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

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
            raise HTTPException(status_code=404, detail=f"No PDF files found in {pdfs_dir}/")
    
    total_chunks = 0
    processed_pdfs = []
    
    for pdf_file in pdf_files:
        try:
            pdf_title = os.path.basename(pdf_file).replace(".pdf", "")
            chunks_count = ingest_pdf(pdf_file, pdf_title)
            total_chunks += chunks_count
            processed_pdfs.append(pdf_title)
        except Exception as e:
            # Continue with other PDFs even if one fails
            print(f"Error ingesting {pdf_file}: {str(e)}")
            continue
    
    return IngestResponse(
        message=f"Successfully ingested {len(processed_pdfs)} PDF(s) with {total_chunks} total chunks",
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
        stats = index.describe_index_stats()
        return {
            "index_name": INDEX_NAME,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# --- Example usage ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

