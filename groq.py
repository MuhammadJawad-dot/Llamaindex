from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
load_dotenv()
llm = GoogleGenAI(
    model="gemini-3.5-flash-lite"
)


embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.embed_model = embed_model
Settings.llm=llm
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
vector_index = VectorStoreIndex.from_documents(documents)
query_engine = vector_index.as_query_engine()
response = query_engine.query(
    "What framework is used for building APIs with Python?"
)

print(response)