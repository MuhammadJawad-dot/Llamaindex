#"gemini-3.5-flash-litepip install python-dotenv
from llama_index.llms.google_genai import GoogleGenAI
from dotenv import load_dotenv
load_dotenv()
llm = GoogleGenAI(
    model="gemini-3.5-flash-lite"
)

response = llm.complete("Explain RAG in one sentence.")

print(response)