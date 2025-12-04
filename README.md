
# RAG Chatbot (Streamlit)

Run an in-memory RAG chatbot without any database. Upload files or paste text, and chat with Google Gemini powered responses.

## Setup

1. Create a `.env` with your `GOOGLE_API_KEY`.
2. Install deps (e.g. with `uv pip install -r pyproject.toml` or `pip install -e .`).

## Run

```bash
streamlit run streamlit_app.py
```

Upload `.txt`, `.md`, or `.pdf` files, or paste text in the sidebar. The app indexes everything in memory with FAISS; nothing is persisted.
