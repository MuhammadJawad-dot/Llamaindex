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
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Modules for Query Transformation
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine, RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever

load_dotenv()

# Setup settings
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.llm = GoogleGenAI(
    model="gemini-3.5-flash-lite"
)

def setup_index():
    print("Loading documents and setting up base index...")
    loader = PyMuPDFReader()
    documents = loader.load_data(file_path="./Chunking_Techniques.pdf")
    
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    
    return VectorStoreIndex(nodes)

def run_hyde_transformation(index, query):
    print("\n" + "="*50)
    print("TECHNIQUE 1: HyDE (Hypothetical Document Embeddings)")
    print("="*50)
    # 1. Base Query Engine
    base_query_engine = index.as_query_engine(similarity_top_k=3)
    
    # 2. Setup HyDE Transform
    # HyDE will first use the LLM to write a hypothetical answer to the query, 
    # then use the embedding of THAT hypothetical answer to search the vector database.
    print("Configuring HyDE Transform...")
    hyde = HyDEQueryTransform(include_original=True, llm=Settings.llm)
    
    # 3. Create the TransformQueryEngine
    hyde_query_engine = TransformQueryEngine(
        base_query_engine, 
        query_transform=hyde
    )
    
    print(f"Querying: {query}")
    response = hyde_query_engine.query(query)
    
    print(f"\nResponse:\n{response}")
    print("\nSource nodes retrieved via HyDE:")
    for i, node in enumerate(response.source_nodes):
        print(f"[{i+1}] Score: {node.score:.4f} | Text: {node.node.get_content().strip()[:100]}...")


def run_multi_query_transformation(index, query):
    print("\n" + "="*50)
    print("TECHNIQUE 2: Multi-Query Transformation")
    print("="*50)
    # 1. Base Retriever
    base_retriever = index.as_retriever(similarity_top_k=3)
    
    # 2. Setup QueryFusionRetriever for Multi-Query
    # By setting num_queries=4, we tell LlamaIndex to use the LLM to generate 
    # 3 additional, slightly reworded variations of our original query.
    # It retrieves documents for ALL 4 queries, and combines the results.
    print("Configuring Multi-Query Fusion Retriever...")
    multi_query_retriever = QueryFusionRetriever(
        [base_retriever],
        similarity_top_k=5,
        num_queries=4,  # Generate additional queries!
        mode="reciprocal_rerank",
        use_async=False # Set to false to avoid async loop issues in simple scripts
    )
    
    # 3. Build Query Engine
    multi_query_engine = RetrieverQueryEngine.from_args(
        retriever=multi_query_retriever,
        llm=Settings.llm
    )
    
    print(f"Querying: {query}")
    response = multi_query_engine.query(query)
    
    print(f"\nResponse:\n{response}")
    print("\nSource nodes retrieved via Multi-Query (RRF Scored):")
    for i, node in enumerate(response.source_nodes):
        score_str = f"{node.score:.4f}" if node.score is not None else "N/A"
        print(f"[{i+1}] Score: {score_str} | Text: {node.node.get_content().strip()[:100]}...")


if __name__ == "__main__":
    index = setup_index()
    
    # A query that might be slightly vague or benefit from expansion
    test_query = "What happens if a sentence is too long to chunk?"
    
    run_hyde_transformation(index, test_query)
    run_multi_query_transformation(index, test_query)
