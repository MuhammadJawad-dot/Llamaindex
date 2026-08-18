import os
import logging
import warnings

# Suppress all warnings and info logs
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)

from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# Setup settings
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.llm = GoogleGenAI(
    model="gemini-3.5-flash-lite"
)

def run_reranking_pipeline():
    print("Loading documents...")
    loader = PyMuPDFReader()
    documents = loader.load_data(file_path="./Chunking_Techniques.pdf")
    
    # 1. Chunk documents into nodes
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    
    # 2. Create vector index
    print("Creating vector index...")
    index = VectorStoreIndex(nodes)
    
    # 3. Configure LLM Reranker
    # We will retrieve the top 5 most similar documents from the vector store,
    # and then use the LLM to rerank them and pick the top 2 best matches.
    print("Configuring LLM Reranker...")
    reranker = LLMRerank(
        choice_batch_size=5, 
        top_n=2, 
        llm=Settings.llm
    )
    
    # 4. Create Query Engine with postprocessors
    query_engine = index.as_query_engine(
        similarity_top_k=5, 
        node_postprocessors=[reranker]
    )
    
    # 5. Run a query
    query = "What are the common chunking techniques?"
    print(f"\nQuerying: {query}")
    print("Retrieving top 5 and reranking to top 2...")
    
    response = query_engine.query(query)
    
    print(f"\nResponse:\n{response}")
    
    # Inspect the source nodes to see the scores assigned by the reranker
    print("\nSource nodes used for final response:")
    for i, node in enumerate(response.source_nodes):
        print(f"[{i+1}] Score: {node.score:.4f} | Text: {node.node.get_content().strip()[:100]}...")

if __name__ == "__main__":
    run_reranking_pipeline()
