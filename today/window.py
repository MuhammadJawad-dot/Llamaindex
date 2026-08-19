import os
from dotenv import load_dotenv

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

load_dotenv()

# Initialize LLM and Embedding Model
llm = GoogleGenAI(model="gemini-3.5-flash-lite")
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

Settings.llm = llm
Settings.embed_model = embed_model

# Create a sample document
text = (
    "LlamaIndex is a data framework for LLM applications. "
    "It provides tools for ingesting, structuring, and accessing private or domain-specific data. "
    "One advanced technique is Sentence Window Retrieval. "
    "This technique parses documents into single sentences, but stores the surrounding context. "
    "When a query is made, the single sentence is retrieved based on embedding similarity. "
    "Then, the sentence is replaced with its surrounding context window before passing it to the LLM. "
    "This provides the LLM with more context while maintaining high precision in retrieval."
)
documents = [Document(text=text)]

# Set up SentenceWindowNodeParser
# This will parse documents into single sentences and add a window of surrounding sentences to metadata
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=2, # Number of sentences on each side of a sentence to capture
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# Extract nodes from the documents
nodes = node_parser.get_nodes_from_documents(documents)

# Build the VectorStoreIndex
index = VectorStoreIndex(nodes)

# Set up the postprocessor to replace the sentence with its window
postprocessor = MetadataReplacementPostProcessor(
    target_metadata_key="window"
)

# Create the query engine
# We pass the postprocessor so that after retrieval, the node text is swapped with the window
query_engine = index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[postprocessor]
)

# Execute a query
query = "What happens to the sentence after it is retrieved?"
response = query_engine.query(query)

print(f"Query: {query}")
print(f"\nResponse: {response}\n")

# Show the source nodes to verify window replacement
if response.source_nodes:
    print("--- Retrieved Source Node (After Window Replacement) ---")
    print(response.source_nodes[0].node.get_content())
