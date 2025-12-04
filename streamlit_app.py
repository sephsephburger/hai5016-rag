import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import get_text_content
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Agent Chat",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Agent Chat")
st.markdown("Ask questions about the blog posts and get AI-powered answers!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the agent (cached to avoid reinitializing on every interaction)
@st.cache_resource
def initialize_agent():
    """Initialize the RAG agent with vector store and tools."""
    
    # Check for API key
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("Please set the GOOGLE_API_KEY environment variable in your .env file.")
        st.stop()
    
    # Initialize model and embeddings
    model = init_chat_model("google_genai:gemini-2.5-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Check for Supabase connection string
    if not os.environ.get("SUPABASE_CONNECTION_STRING"):
        st.error("Please set the SUPABASE_CONNECTION_STRING environment variable in your .env file.")
        st.stop()
    
    # Initialize vector store
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="lilianweng_blog",
        connection=os.environ["SUPABASE_CONNECTION_STRING"],
    )
    
    # Define the retrieval tool
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    
    # Create the agent
    tools = [retrieve_context]
    prompt = (
        "You are a helpful AI assistant with access to a tool that retrieves context from blog posts. "
        "Use the tool to help answer user queries accurately. "
        "Always cite sources when providing information from the retrieved context."
    )
    agent = create_agent(model, tools, system_prompt=prompt)
    
    return agent

# Initialize the agent
try:
    agent = initialize_agent()
except Exception as e:
    st.error(f"Error initializing agent: {str(e)}")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the blog posts..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Stream the agent's response
            for event in agent.stream(
                {"messages": [{"role": "user", "content": prompt}]},
                stream_mode="values",
            ):
                last_message = event["messages"][-1]
                
                # Extract text content (handles both string and list payloads)
                content = get_text_content(last_message)
                if content:
                    full_response = content
                    message_placeholder.markdown(full_response)
            
            # Display final response
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            message_placeholder.error(error_message)
            full_response = error_message
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar with info and controls
with st.sidebar:
    st.header("About")
    st.markdown("""
    This is a RAG (Retrieval-Augmented Generation) agent that can answer questions 
    about blog posts stored in a vector database.
    
    **Features:**
    - 🔍 Semantic search through blog content
    - 🤖 AI-powered responses using Google Gemini
    - 💬 Interactive chat interface
    """)
    
    st.divider()
    
    # Clear chat button
    if st.button("Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    st.caption("Powered by LangChain, Google Gemini & Supabase")
