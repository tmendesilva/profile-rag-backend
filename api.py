from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from prompt import rag, rag_stream
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="RAG API")

# Add CORS middleware to allow frontend to consume the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tmendesilva.github.io", "http://localhost:3000"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- 2. SECURITY CONFIGURATION ---
# This looks for an "Authorization" header in the incoming request
api_key_header = APIKeyHeader(name='Authorization', auto_error=True)

# Dependency function to validate the token
async def validate_token(token: str = Depends(api_key_header)):
    if token != os.getenv('API_KEY'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials. Invalid Token."
        )
    return token

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"

class QueryResponse(BaseModel):
    answer: str

@app.post("/query", dependencies=[Depends(validate_token)])
async def query_rag(request: QueryRequest):
    """Query the RAG system with a user question (streaming)"""
    try:
        return StreamingResponse(rag_stream(request.query, request.session_id), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-sync", dependencies=[Depends(validate_token)])
async def query_rag_sync(request: QueryRequest):
    """Query the RAG system with a user question (non-streaming)"""
    try:
        answer = rag(request.query, request.session_id)
        return PlainTextResponse(content=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "RAG API is running", "endpoints": {"/query": "POST - Query the RAG system"}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    