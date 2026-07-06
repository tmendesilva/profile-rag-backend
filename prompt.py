import os
import sys

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="retrieval_query",
    google_api_key=GEMINI_API_KEY
)

# Load the persisted database
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Define the state with context and messages
class RAGState(TypedDict):
    contexto: str
    pergunta: str
    messages: list

# Create the LangGraph with checkpointer
memory = MemorySaver()

def call_model(state: RAGState):
    # Build the system prompt with context
    system_prompt = f"""Answer the user query based on context. If you dont't know the answer or the context do not have the answer, say that you don't know.
Always answer using the same language as the user's question.
You are going to answer questions about Tiago's life. Answer as if you were Tiago.
Answer only questions related to Tiago's life, If the question is not related to Tiago's professional life, say that you can't answer that question.

## CONTEXT
{state['contexto']}"""
    
    # Create messages with system prompt, conversation history, and the user's question
    messages = [SystemMessage(content=system_prompt)]
    # Add existing conversation history
    if state.get("messages"):
        messages.extend(state["messages"])
    # Add the current user question
    messages.append(HumanMessage(content=state["pergunta"]))
    
    # Call the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=GEMINI_API_KEY)
    response = llm.invoke(messages)
    
    # Extract text content from response (handle both string and list formats)
    content = response.content
    if isinstance(content, list):
        content = content[0].get('text', str(content)) if content else str(content)
    
    # Return the new message to append to history
    return {"messages": [AIMessage(content=content)]}

workflow = StateGraph(RAGState)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

app = workflow.compile(checkpointer=memory)

def format_context(documents):
    """Format documents with metadata for better context"""
    formatted = []
    for doc in documents:
        metadata = doc.metadata
        content = doc.page_content
        
        # Add section context from metadata
        section_info = ""
        if "Header2" in metadata:
            section_info = f"[Seção: {metadata['Header2']}]"
        if "Header3" in metadata:
            section_info += f" [Subseção: {metadata['Header3']}]"
        
        formatted.append(f"{section_info}\n{content}\n")
    
    return "\n---\n".join(formatted)

def rag(user_query: str, session_id: str = "default") -> str:
    context = db.similarity_search(user_query, k=5)
    formatted_context = format_context(context)
    config = {"configurable": {"thread_id": session_id}}

    # Get current state to include existing messages
    current_state = app.get_state(config)
    existing_messages = current_state.values.get("messages", []) if current_state else []

    response = app.invoke(
        {"contexto": formatted_context, "pergunta": user_query, "messages": existing_messages},
        config
    )
    
    # Get the last AI message from messages
    if response["messages"]:
        last_message = response["messages"][-1]
        if isinstance(last_message, AIMessage):
            content = last_message.content
            # Handle both string and list formats
            if isinstance(content, list):
                content = content[0].get('text', str(content)) if content else str(content)
            return content
    return ""

def rag_stream(user_query: str, session_id: str = "default"):
    context = db.similarity_search(user_query, k=5)
    formatted_context = format_context(context)
    config = {"configurable": {"thread_id": session_id}}
    
    # Get current state to include existing messages
    current_state = app.get_state(config)
    existing_messages = current_state.values.get("messages", []) if current_state else []
    
    # Stream the response
    for event in app.stream(
        {"contexto": formatted_context, "pergunta": user_query, "messages": existing_messages},
        config
    ):
        for node_name, node_output in event.items():
            if "messages" in node_output and node_output["messages"]:
                last_message = node_output["messages"][-1]
                if isinstance(last_message, AIMessage):
                    content = last_message.content
                    # Handle both string and list formats
                    if isinstance(content, list):
                        content = content[0].get('text', str(content)) if content else str(content)
                    yield f"data: {content}\n\n"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python prompt.py 'your question here'")
        sys.exit(1)
    
    user_query = sys.argv[1]
    print(rag(user_query))