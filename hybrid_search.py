import os
import logging
import warnings

# Suppress all warnings and info logs
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)

from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex, SimpleKeywordTableIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

load_dotenv()

# Setup settings
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.llm = GoogleGenAI(
    model="gemini-3.5-flash-lite"
)

def run_hybrid_pipeline():
    print("Loading documents...")
    loader = PyMuPDFReader()
    documents = loader.load_data(file_path="./Chunking_Techniques.pdf")
    
    # 1. Chunk documents
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    
    # 2. Setup Vector Retriever (Dense/Semantic Search)
    print("Setting up Vector Retriever...")
    vector_index = VectorStoreIndex(nodes)
    vector_retriever = vector_index.as_retriever(similarity_top_k=5)
    
    # 3. Setup Keyword Retriever (Sparse Search)
    # We use SimpleKeywordTableIndex which uses simple regex to find keywords 
    # and doesn't require C++ build tools like BM25 does on Windows.
    print("Setting up Keyword Retriever...")
    keyword_index = SimpleKeywordTableIndex(nodes)
    keyword_retriever = keyword_index.as_retriever(retriever_mode="simple")
    
    # 4. Combine into a Hybrid Retriever
    # QueryFusionRetriever takes multiple retrievers, runs them, and combines the 
    # results using Reciprocal Rank Fusion (RRF). 
    print("Configuring Hybrid Query Fusion Retriever...")
    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, keyword_retriever],
        similarity_top_k=5,
        num_queries=1,  # Set to 1 to just run the exact original query
        mode="reciprocal_rerank", # Combine scores using RRF
        use_async=False # Set to false to avoid async loop issues in simple scripts
    )
    
    # 5. Build a Query Engine using our custom hybrid retriever
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        llm=Settings.llm
    )
    
    query = "What are the common chunking techniques?"
    print(f"\nQuerying: {query}")
    print("Running Hybrid Search (Vector + Keyword)...")
    
    response = query_engine.query(query)
    
    print(f"\nResponse:\n{response}")
    
    print("\nSource nodes retrieved via Hybrid Search (RRF Scored):")
    for i, node in enumerate(response.source_nodes):
        # We handle displaying the score nicely even if it's None
        score_str = f"{node.score:.4f}" if node.score is not None else "N/A"
        print(f"[{i+1}] Score: {score_str} | Text: {node.node.get_content().strip()[:100]}...")

if __name__ == "__main__":
    run_hybrid_pipeline()
