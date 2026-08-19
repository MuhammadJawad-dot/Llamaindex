import os
from dotenv import load_dotenv

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline

load_dotenv()

# Initialize LLM
llm = GoogleGenAI(model="gemini-3.5-flash-lite")
Settings.llm = llm

# Create sample document
text = (
    "LlamaIndex is an open-source framework designed to connect custom data sources to large language models (LLMs). "
    "It provides a comprehensive toolset including data connectors, node parsers, and various indexing strategies. "
    "A key feature is its ability to extract automatic metadata during ingestion, which heavily improves retrieval accuracy. "
    "By attaching questions that a chunk can answer, or summarizing the content, semantic search becomes much more precise."
)
documents = [Document(text=text, metadata={"author": "LlamaIndex Docs"})]

# Set up metadata extractors
# These will use the LLM to automatically generate and attach metadata to each chunk
extractors = [
    TitleExtractor(nodes=5, llm=llm),
    QuestionsAnsweredExtractor(questions=3, llm=llm)
]

# Create an Ingestion Pipeline
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=64, chunk_overlap=16),
        *extractors
    ]
)

# Run the pipeline to process documents into nodes and extract metadata
print("Running metadata extraction pipeline... (This makes LLM calls)")
nodes = pipeline.run(documents=documents)

# Inspect the extracted metadata
print("\n--- Extracted Nodes and Metadata ---\n")
for idx, node in enumerate(nodes):
    print(f"Node {idx + 1} Text:")
    print(node.get_content())
    print("\nMetadata:")
    for key, value in node.metadata.items():
        print(f"  - {key}: {value}")
    print("-" * 50)
