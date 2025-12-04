import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
import os


def _normalize_content(content):
    """Return text content from LangChain message content, handling list payloads."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _render_thinking_log(log):
    """Render a readable chain-of-thought style trace."""
    if not log:
        return "Waiting for reasoning steps..."
    lines = []
    for idx, entry in enumerate(log, start=1):
        lines.append(f"**Step {idx}:** {entry}")
    return "\n\n".join(lines)

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

# Sidebar status placeholder
status_placeholder = st.sidebar.empty()
status_placeholder.info("Status: Idle")

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
    
    return agent, vector_store

# Initialize the agent
try:
    agent, vector_store = initialize_agent()
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
    status_placeholder.warning("Status: Thinking...")
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        with st.expander("See Thinking Process", expanded=False):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("Waiting for reasoning steps...")
        full_response = ""
        thinking_log = []
        sources = []
        
        try:
            # Pre-fetch citations to render numbered pills
            try:
                sources = vector_store.similarity_search(prompt, k=3)
            except Exception:
                sources = []
            
            # Stream the agent's response
            for event in agent.stream(
                {"messages": [{"role": "user", "content": prompt}]},
                stream_mode="values",
            ):
                last_message = event.get("messages", [])[-1] if event.get("messages") else None
                
                # Extract text content (handles both string and list payloads)
                content = _normalize_content(getattr(last_message, "content", "")) if last_message else ""
                if content:
                    full_response = content

                # Append event trace for thinking log (truncated for readability)
                if event.get("messages"):
                    entries = []
                    for msg in event["messages"]:
                        role = getattr(msg, "role", "unknown")
                        text = _normalize_content(getattr(msg, "content", ""))
                        if text:
                            text = text[:300]
                        entries.append(f"{role}: {text}")
                    thinking_log.append(" · ".join(entries))
                    thinking_placeholder.markdown(_render_thinking_log(thinking_log))
            
            # Display final response
            citation_markup = ""
            if sources:
                pills = []
                for idx, doc in enumerate(sources, start=1):
                    label = (
                        "<span style='display:inline-block;padding:2px 8px;"
                        "border-radius:999px;border:1px solid #ccc;font-size:0.75rem;"
                        "margin-right:6px;background-color:#f5f5f5;'>"
                        f"{idx}</span>"
                    )
                    meta = doc.metadata if doc.metadata else {}
                    source_text = meta if isinstance(meta, str) else meta.get("source", meta)
                    pills.append(f"{label}<span style='font-size:0.85rem;'>{source_text}</span>")
                citation_markup = "<br/><br/><div><strong>Sources</strong>: " + " ".join(pills) + "</div>"

            message_placeholder.markdown(full_response + citation_markup, unsafe_allow_html=True)
            status_placeholder.success("Status: Done")
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            message_placeholder.error(error_message)
            full_response = error_message
            status_placeholder.error("Status: Error")
        
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
