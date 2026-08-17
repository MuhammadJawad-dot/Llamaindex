from llama_index.core import SimpleDirectoryReader,Settings,VectorStoreIndex
from llama_index.llms. google_genai import GoogleGenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
from dotenv import load_dotenv
import os
load_dotenv()

# 1. Embedding model
# Settings.embed_model = HuggingFaceInferenceAPIEmbedding(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     token=os.getenv("huggingface")
# )
Settings.embed_model=HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Settings.llm=GoogleGenAI(
#     model="gemini-3.5-flash-lite"
# )
# 2. Load documents
documents = SimpleDirectoryReader(
    input_files=["./Attention.pdf"]
).load_data()


# 3. Chunk documents into nodes
splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)

nodes = splitter.get_nodes_from_documents(documents)


# 4. Create vector index
index = VectorStoreIndex(nodes)


print(f"Documents: {len(documents)}")
print(f"Nodes: {len(nodes)}")