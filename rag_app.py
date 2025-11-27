# %%
from langchain.chat_models import init_chat_model
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

# %%
if not os.environ.get("GOOGLE_API_KEY"):
  print("Please set the GOOGLE_API_KEY environment variable.")

model = init_chat_model("google_genai:gemini-2.5-flash")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# %%
if not os.environ.get("SUPABASE_CONNECTION_STRING"):
    os.environ["SUPABASE_CONNECTION_STRING"] = getpass.getpass("Enter connection string for Supabase: ")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="lilianweng_blog",
    connection=os.environ["SUPABASE_CONNECTION_STRING"],
)

# %%
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# %%
from langchain.agents import create_agent


tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)

# %%
query = (
    "What is the standard method for Task Decomposition?\n\n"
    "Once you get the answer, look up common extensions of that method."
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    last_message = event["messages"][-1]
    last_message.pretty_print()

# %%



