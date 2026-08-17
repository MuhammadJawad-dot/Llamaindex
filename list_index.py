from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import SummaryIndex # Note: ListIndex was renamed to SummaryIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize the Google GenAI LLM
llm = GoogleGenAI(
    model="gemini-3.5-flash-lite"
)

# 2. Initialize the Embedding model
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Set both the LLM and Embedding model in Settings
Settings.llm = llm
Settings.embed_model = embed_model

documents = [
    Document(
        text="Python is a high-level programming language used for web development, automation, and data science."
    ),
    Document(
        text="FastAPI is a modern Python framework used for building APIs."
    ),
    Document(
        text="LlamaIndex is a framework for building applications that work with large amounts of data and LLMs."
    ),
    Document(
        text="LangGraph is used to build stateful and controllable AI agent workflows."
    ),
]

# 4. Create the SummaryIndex (formerly ListIndex)
# This simply stores nodes as a sequential list rather than using a vector database representation.
summary_index = SummaryIndex.from_documents(documents)

# 5. Create the query engine
query_engine = summary_index.as_query_engine()

# 6. Query the index
response = query_engine.query(
    "What framework is used for building APIs with Python?"
)

print(response)
