import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Prefer community FAISS; fall back to legacy import
try:
    from langchain_community.vectorstores import FAISS
except ImportError:  # pragma: no cover
    try:
        from langchain.vectorstores import FAISS  # type: ignore
    except ImportError as exc:
        raise ImportError("FAISS vector store not available. Install langchain-community.") from exc

# Support both the standalone splitter package and older langchain splitters.
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - Streamlit Cloud may miss the extra package
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "RecursiveCharacterTextSplitter not found. Install langchain-text-splitters>=0.3.0."
        ) from exc


# Load environment variables early so Streamlit picks them up
load_dotenv()


st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="wide",
)

st.title("RAG Chatbot (no database)")
st.caption("Bring your own files, we keep everything in memory.")


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the chat model."""
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("Missing GOOGLE_API_KEY. Add it to your .env before running.")
        st.stop()
    return init_chat_model("google_genai:gemini-2.5-flash")


@st.cache_resource(show_spinner=False)
def load_embeddings():
    """Load embeddings model."""
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("Missing GOOGLE_API_KEY. Add it to your .env before running.")
        st.stop()
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def build_documents(files: List, pasted_text: str) -> List[Document]:
    """Convert uploads and pasted text into LangChain Documents."""
    docs: List[Document] = []

    for file in files or []:
        suffix = Path(file.name).suffix.lower()

        if suffix in {".txt", ".md"}:
            content = file.read().decode("utf-8", errors="ignore")
            docs.append(Document(page_content=content, metadata={"source": file.name}))

        elif suffix == ".pdf":
            # Persist to a temp file for pypdf
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getbuffer())
                temp_path = tmp.name
            try:
                from pypdf import PdfReader  # lazy import to avoid hard dependency at app start
            except ImportError:
                st.error("PDF support needs the 'pypdf' package. Add it to requirements and redeploy.")
                os.unlink(temp_path)
                continue
            reader = PdfReader(temp_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": file.name, "page": i + 1},
                    )
                )
            os.unlink(temp_path)

        else:
            st.warning(f"Unsupported file type skipped: {file.name}")

    if pasted_text and pasted_text.strip():
        docs.append(Document(page_content=pasted_text.strip(), metadata={"source": "pasted_text"}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.split_documents(docs)


def build_vector_store(docs: List[Document]):
    """Embed docs into an in-memory FAISS index."""
    embeddings = load_embeddings()
    return FAISS.from_documents(docs, embeddings)


def render_sidebar() -> Tuple[List[Document], bool]:
    """Handle uploads, pasted text, and vector store creation."""
    with st.sidebar:
        st.header("Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload files (.txt, .md, .pdf)",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
        )
        pasted_text = st.text_area("Or paste text", height=180, placeholder="Paste notes, blog posts, etc.")

        build_clicked = st.button("Build vector store", type="primary")
        reset_clicked = st.button("Clear vector store", type="secondary")

    docs: List[Document] = []
    if build_clicked:
        docs = build_documents(uploaded_files or [], pasted_text)
        if not docs:
            st.warning("No ingestible content found. Add text or files first.")
        else:
            with st.spinner("Indexing content in-memory..."):
                st.session_state.vector_store = build_vector_store(docs)
                st.session_state.kb_sources = sorted({doc.metadata.get("source", "unknown") for doc in docs})
                st.success(f"Indexed {len(docs)} chunks from {len(st.session_state.kb_sources)} source(s).")

    if reset_clicked:
        st.session_state.vector_store = None
        st.session_state.kb_sources = []
        st.success("Cleared vector store.")

    return docs, build_clicked


def retrieve_context(query: str):
    """Fetch top matching chunks with the in-memory index."""
    vector_store = st.session_state.get("vector_store")
    if vector_store is None:
        st.warning("Add documents and build the vector store before chatting.")
        st.stop()
    return vector_store.similarity_search(query, k=4)


def stream_answer(question: str, context_docs: List[Document]) -> str:
    """Generate an answer with streaming updates."""
    model = load_model()

    context_text = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in context_docs
    )
    system_prompt = (
        "You are a helpful chatbot that only answers using the provided context.\n"
        "If the context does not contain the answer, say you do not know.\n"
        "Include brief source labels that support your answer.\n\n"
        f"Context:\n{context_text}"
    )

    full_response = ""
    message_placeholder = st.empty()

    for chunk in model.stream([SystemMessage(content=system_prompt), HumanMessage(content=question)]):
        full_response += chunk.content or ""
        message_placeholder.markdown(full_response)

    sources = sorted({doc.metadata.get("source", "unknown") for doc in context_docs})
    if sources:
        full_response += f"\n\n_Sources: {', '.join(sources)}_"
        message_placeholder.markdown(full_response)

    return full_response


def init_state():
    """Ensure session state keys exist."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("vector_store", None)
    st.session_state.setdefault("kb_sources", [])


def main():
    init_state()
    render_sidebar()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask anything about your uploaded content...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve and answer
        with st.chat_message("assistant"):
            context_docs = retrieve_context(prompt)
            answer = stream_answer(prompt, context_docs)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Footer
    st.sidebar.divider()
    st.sidebar.caption(
        "In-memory RAG chatbot. Nothing is persisted; reload the vector store each run."
    )


if __name__ == "__main__":
    main()
