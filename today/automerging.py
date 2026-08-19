import os
from dotenv import load_dotenv

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Document, Settings, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

load_dotenv()

# Initialize LLM and Embedding Model
llm = GoogleGenAI(model="gemini-3.5-flash-lite")
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

Settings.llm = llm
Settings.embed_model = embed_model

# Create a sample document long enough to be chunked into multiple pieces
text = (
    "LlamaIndex is a data framework for LLM applications. "
    "It provides tools for ingesting, structuring, and accessing private or domain-specific data. "
    "One advanced technique is Auto-Merging Retrieval. "
    "This technique parses documents into a hierarchy of nodes, such as large parent nodes and smaller child nodes. "
    "When a query is made, the retriever fetches the most relevant small child nodes based on embedding similarity. "
    "If a certain threshold of child nodes from the same parent are retrieved, it automatically merges them. "
    "The merged parent node is then returned to provide more complete context to the LLM. "
    "This maintains high precision while preventing fragmented context windows."
)
documents = [Document(text=text)]

# Set up HierarchicalNodeParser
# This will parse documents into a hierarchy of nodes (parent, child, grandchild)
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[256, 128, 64] # Define the sizes of the chunks in the hierarchy
)

# Extract nodes from the documents
nodes = node_parser.get_nodes_from_documents(documents)
# We only want to embed and index the smallest chunks (leaf nodes)
leaf_nodes = get_leaf_nodes(nodes)

# Set up StorageContext to store all nodes (including parents) 
# so the retriever can look them up during the merging phase
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

# Build the VectorStoreIndex using ONLY the leaf nodes for embedding
index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context
)

# Set up the base retriever
base_retriever = index.as_retriever(similarity_top_k=4)

# Set up the AutoMergingRetriever
# It will intercept the leaf nodes and merge them into parent nodes if thresholds are met
retriever = AutoMergingRetriever(
    base_retriever, 
    storage_context, 
    verbose=True
)

# Create the query engine using the auto-merging retriever
query_engine = RetrieverQueryEngine.from_args(retriever)

# Execute a query
query = "Explain how child nodes are handled in this retrieval technique."
response = query_engine.query(query)

print(f"Query: {query}")
print(f"\nResponse: {response}\n")

# Show the source nodes to verify whether they were merged
if response.source_nodes:
    print("--- Retrieved Source Nodes (After Auto-Merging) ---")
    for idx, node in enumerate(response.source_nodes):
        print(f"Node {idx + 1}:")
        print(node.node.get_content())
        print("-" * 20)
