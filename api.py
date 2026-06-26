from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from prompt import rag, rag_stream

app = FastAPI(title="RAG API")

# Add CORS middleware to allow frontend to consume the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tmendesilva.github.io", "http://localhost:3000"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"

class QueryResponse(BaseModel):
    answer: str

@app.post("/query")
async def query_rag(request: QueryRequest):
    """Query the RAG system with a user question (streaming)"""
    try:
        return StreamingResponse(rag_stream(request.query, request.session_id), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-sync")
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
