from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

logger.info("Starting database initialization...")

# Load markdown file
logger.info("Loading profile.md...")
with open("assets/profile.md", "r", encoding="utf-8") as f:
    texto = f.read()
logger.info(f"Loaded profile.md: {len(texto)} characters")

# Create documents
logger.info("Creating documents...")
documents = [
    Document(
        page_content=texto,
        metadata={
            "author": "Tiago Mendes", 
            "source": "https://www.linkedin.com/in/tmendesilva/"
        }
    )
]
logger.info(f"Created {len(documents)} document(s)")

# Split text into chunks
logger.info("Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=250,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)
logger.info(f"Split into {len(chunks)} chunks")

# Create embeddings
logger.info("Creating embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="retrieval_document",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
logger.info("Embeddings model initialized")

# Create Chroma database
logger.info("Creating Chroma database...")
db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings,
    persist_directory="./chroma_db"
)
logger.info(f"Chroma database created and persisted to ./chroma_db")
logger.info("Database initialization complete!")
