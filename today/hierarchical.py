import os
from dotenv import load_dotenv

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

load_dotenv()

# Initialize LLM and Embedding Model
llm = GoogleGenAI(model="gemini-3.5-flash-lite")
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

Settings.llm = llm
Settings.embed_model = embed_model

# Create sample documents (e.g., representing different topics/files)
doc1 = Document(
    text=(
        "LlamaIndex is a data framework for LLM applications. "
        "It helps with ingestion, structuring, and retrieval of custom data. "
        "It supports many vector databases and LLM providers."
    ),
    metadata={"title": "LlamaIndex Overview"}
)

doc2 = Document(
    text=(
        "Hierarchical retrieval is a technique where you retrieve a summary node first, "
        "and then recursively retrieve the detailed chunks associated with that summary. "
        "This is useful for complex queries that span across multiple long documents."
    ),
    metadata={"title": "Retrieval Techniques"}
)

# Step 1: Parse the documents into chunks
splitter = SentenceSplitter(chunk_size=64, chunk_overlap=16)
doc1_nodes = splitter.get_nodes_from_documents([doc1])
doc2_nodes = splitter.get_nodes_from_documents([doc2])

# Step 2: Create a separate VectorStoreIndex for each document's chunks
doc1_index = VectorStoreIndex(doc1_nodes)
doc2_index = VectorStoreIndex(doc2_nodes)

# Define unique IDs for each sub-index
doc1_id = "doc1_index"
doc2_id = "doc2_index"

# Step 3: Create IndexNodes that summarize each document and point to the sub-indices
# These act as the "top level" of our hierarchy
summary1 = IndexNode(
    text="This document covers an overview of the LlamaIndex framework, including ingestion and vector DBs.",
    index_id=doc1_id
)
summary2 = IndexNode(
    text="This document discusses advanced retrieval techniques like hierarchical and recursive retrieval.",
    index_id=doc2_id
)

# Step 4: Create a top-level index over the IndexNodes
top_level_index = VectorStoreIndex([summary1, summary2])

# Step 5: Configure the retrievers
top_level_retriever = top_level_index.as_retriever(similarity_top_k=1)

# A dictionary mapping index_ids to their respective retrievers
retriever_dict = {
    doc1_id: doc1_index.as_retriever(similarity_top_k=2),
    doc2_id: doc2_index.as_retriever(similarity_top_k=2),
}

# Step 6: Create the RecursiveRetriever
# It will use the top-level retriever first. If it retrieves an IndexNode, 
# it will use the index_id to fetch the associated sub-retriever and recursively query it.
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": top_level_retriever, **retriever_dict},
    verbose=True
)

# Create the query engine
query_engine = RetrieverQueryEngine.from_args(recursive_retriever)

# Execute a query that targets the second document
query = "What is hierarchical retrieval useful for?"
response = query_engine.query(query)

print(f"Query: {query}")
print(f"\nResponse: {response}\n")

# Show the final source nodes retrieved to answer the query
if response.source_nodes:
    print("--- Final Retrieved Source Nodes (After Recursive Retrieval) ---")
    for idx, node in enumerate(response.source_nodes):
        print(f"Node {idx + 1}:")
        print(node.node.get_content())
        print("-" * 20)
