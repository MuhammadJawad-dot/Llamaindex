import os
import logging
import warnings

# Suppress all warnings and info logs FIRST, before importing anything else!
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)

from llama_index.core import SimpleDirectoryReader,Settings,VectorStoreIndex
from llama_index.llms. google_genai import GoogleGenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
from llama_index.readers.file import PyMuPDFReader
from dotenv import load_dotenv
# from llama_parse import LlamaParse

load_dotenv()

# 1. Embedding model
# Settings.embed_model = HuggingFaceInferenceAPIEmbedding(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     token=os.getenv("huggingface")
# )
Settings.embed_model=HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.llm=GoogleGenAI(
    model="gemini-3.5-flash-lite"
)
# 2. Load documents
# parser = LlamaParse(result_type="markdown")
# documents = parser.load_data("./Chunking_Techniques.pdf")

loader = PyMuPDFReader()
documents = loader.load_data(file_path="./Chunking_Techniques.pdf")
# print(documents[0].text)
# 3. Chunk documents into nodes
splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)

nodes = splitter.get_nodes_from_documents(documents)


# 4. Create vector index
index = VectorStoreIndex(nodes)
query_engine=index.as_query_engine()
respone=query_engine.query("What is chunking?")

# print(f"Documents: {len(documents)}")
# print(f"Nodes: {len(nodes)}")
print(f"Response: {respone}")