from typing import List, Sequence, Any
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import BaseNode, Document, TextNode
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

load_dotenv()

# --- 1. Define the Custom Parser ---
class ParagraphSplitter(NodeParser):
    """
    A custom node parser that simply splits documents into chunks 
    based on paragraphs (double newlines).
    """
    
    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        parsed_nodes = []
        
        for source_node in nodes:
            # Custom splitting logic: split the text by double newlines
            paragraphs = source_node.get_content().split("\n\n")
            
            for p in paragraphs:
                p = p.strip()
                if p:
                    # Create a new TextNode for each chunk of text
                    new_node = TextNode(text=p)
                    
                    # Important: inherit metadata from the source document
                    new_node.metadata = source_node.metadata.copy()
                    
                    parsed_nodes.append(new_node)
                    
        return parsed_nodes


# --- 2. Setup Models ---
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.llm = GoogleGenAI(
    model="gemini-3.5-flash-lite"
)

# --- 3. Create a Sample Document ---
documents = [
    Document(
        text="""This is the first paragraph of the document.
It has multiple sentences.

This is the second paragraph. It should become its own separate node when using our custom parser!

And finally, this is the third paragraph."""
    )
]

# --- 4. Use the Custom Parser ---
print("Running custom parser...")
custom_parser = ParagraphSplitter()

# Parse the documents into nodes
nodes = custom_parser.get_nodes_from_documents(documents)

print(f"\nOriginal documents: {len(documents)}")
print(f"Generated nodes (paragraphs): {len(nodes)}")

# Print out the nodes to verify they were split correctly
for i, node in enumerate(nodes):
    print(f"\n--- Node {i+1} ---")
    print(node.text)

# --- 5. Index the custom nodes (just like normal!) ---
# index = VectorStoreIndex(nodes)
