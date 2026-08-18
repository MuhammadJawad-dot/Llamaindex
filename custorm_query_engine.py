import os
import logging
import warnings

# Suppress all warnings and info logs
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)

from dotenv import load_dotenv
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader
from pydantic import Field

load_dotenv()

# Setup Settings
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.llm = GoogleGenAI(
    model="gemini-3.5-flash-lite"
)

# 1. Define a Custom Prompt
QA_PROMPT = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

# 2. Define the Custom Query Engine
class MyCustomQueryEngine(CustomQueryEngine):
    """A custom query engine that uses a basic retriever and prompt."""
    
    retriever: BaseRetriever = Field(description="The retriever to use for fetching context.")
    llm: GoogleGenAI = Field(description="The language model used for synthesis.")
    qa_prompt: PromptTemplate = Field(description="The prompt template for question answering.")

    def custom_query(self, query_str: str):
        # Step 1: Retrieve relevant nodes
        nodes = self.retriever.retrieve(query_str)
        
        # Step 2: Extract text from nodes to form the context
        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        
        # Step 3: Format the prompt with context and query
        formatted_prompt = self.qa_prompt.format(
            context_str=context_str, query_str=query_str
        )
        
        # Step 4: Call the LLM to get the answer
        response = self.llm.complete(formatted_prompt)
        
        return str(response)

if __name__ == "__main__":
    print("Loading documents and creating index...")
    # Load and process the document
    loader = PyMuPDFReader()
    documents = loader.load_data(file_path="./Chunking_Techniques.pdf")
    
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    
    # Create the vector index
    index = VectorStoreIndex(nodes)
    
    # Get the base retriever
    retriever = index.as_retriever(similarity_top_k=2)
    
    # Initialize our custom query engine
    query_engine = MyCustomQueryEngine(
        retriever=retriever,
        llm=Settings.llm,
        qa_prompt=QA_PROMPT
    )
    
    # Run a test query
    query = "What are the common chunking techniques?"
    print(f"\nQuerying: {query}")
    
    # The .query() method is provided by the base class, which under the hood calls our custom_query()
    response = query_engine.query(query)
    
    print(f"\nCustom Query Engine Response:\n{response}")
