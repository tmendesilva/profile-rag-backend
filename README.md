# Profile RAG Backend

A Retrieval-Augmented Generation (RAG) backend API that provides intelligent question-answering about a personal profile. This system uses vector embeddings to retrieve relevant context from a profile document and generates natural language responses using Google's Gemini AI.

## Features

- **Intelligent Q&A**: Answer questions about a personal profile using RAG methodology
- **Streaming Support**: Both streaming and non-streaming API endpoints
- **Conversation Memory**: Session-based conversation history for contextual multi-turn dialogues
- **Context-Aware**: Retrieves relevant document chunks using semantic similarity search
- **Secure API**: API key authentication for secure access
- **CORS Enabled**: Configured for frontend integration
- **LangGraph Orchestration**: Uses LangGraph for stateful conversation management

## Tech Stack

- **Backend Framework**: FastAPI
- **Vector Database**: ChromaDB
- **Embeddings**: Google Generative AI (Gemini Embedding Model)
- **LLM**: Google Gemini 3.1 Flash Lite
- **Orchestration**: LangChain & LangGraph
- **Document Processing**: LangChain Text Splitters
- **Language**: Python 3.x

## Installation

### Prerequisites

- Python 3.8 or higher
- Google Gemini API Key
- A `.env` file with your configuration

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd profile-rag-backend
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
Create a `.env` file in the project root with the following:
```env
GEMINI_API_KEY=your_gemini_api_key_here
API_KEY=your_api_key_for_authentication
```

5. Initialize the vector database:
```bash
python db.py
```
This will load the profile from `assets/profile.md`, split it into chunks, create embeddings, and store them in ChromaDB.

## Usage

### Running the API

Start the FastAPI server:
```bash
python api.py
```

Or use uvicorn directly:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Root Endpoint
```
GET /
```
Returns API status and available endpoints.

#### Query (Streaming)
```
POST /query
Headers:
  Authorization: <your_api_key>
Body:
  {
    "query": "What is Tiago's background?",
    "session_id": "user_session_123"
  }
```
Returns a streaming response with Server-Sent Events (SSE).

#### Query (Non-Streaming)
```
POST /query-sync
Headers:
  Authorization: <your_api_key>
Body:
  {
    "query": "What is Tiago's background?",
    "session_id": "user_session_123"
  }
```
Returns a plain text response with the answer.

### Testing with cURL

```bash
# Non-streaming query
curl -X POST http://localhost:8000/query-sync \
  -H "Authorization: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Tiago experience?", "session_id": "test"}'

# Streaming query
curl -X POST http://localhost:8000/query \
  -H "Authorization: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about Tiago", "session_id": "test"}'
```

### Command Line Testing

You can also test the RAG system directly from the command line:
```bash
python prompt.py "What is Tiago's professional background?"
```

## Project Structure

```
profile-rag-backend/
├── api.py              # FastAPI application and endpoints
├── db.py               # Vector database initialization
├── prompt.py           # RAG logic and LangGraph workflow
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (not committed)
├── .gitignore          # Git ignore rules
├── assets/
│   └── profile.md      # Source profile document
├── chroma_db/          # Vector database storage (persisted)
└── README.md           # This file
```

## Configuration

### Environment Variables

- `GEMINI_API_KEY`: Required. Your Google Gemini API key for embeddings and LLM
- `API_KEY`: Required. Secret key for API authentication

### CORS Configuration

The API is configured to allow requests from:
- `https://tmendesilva.github.io` (production frontend)
- `http://localhost:3000` (development frontend)

Modify the `allow_origins` list in `api.py` to add your frontend domain.

## How It Works

1. **Document Processing**: The profile.md file is split into chunks based on markdown headers
2. **Embedding Creation**: Each chunk is converted to vector embeddings using Google's embedding model
3. **Vector Storage**: Embeddings are stored in ChromaDB for efficient similarity search
4. **Query Processing**: When a question is received:
   - The query is embedded and similar chunks are retrieved from ChromaDB
   - The retrieved context is formatted with section information
   - A system prompt instructs the LLM to answer as if it were Tiago
   - The LLM generates a response based on the retrieved context
5. **Conversation Memory**: LangGraph maintains conversation history per session ID for multi-turn dialogues

## Development

### Code Style

The project uses Ruff for linting and formatting. Configuration is in `ruff.toml`.

### Logging

Logging configuration is in `log_config.yaml`. The system logs initialization steps and errors.

## License

MIT License

Copyright (c) 2024 Profile RAG Backend

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
