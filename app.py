import gradio as gr
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from src.embedder import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import Generator

# Initialize RAG components globally so they don't reload on every query
base_dir = Path(__file__).resolve().parent
db_dir = base_dir / "data" / "chroma_db"

print("[*] Initializing RAG Backend for Frontend...")
try:
    vector_store = VectorStore(persist_directory=str(db_dir))
    vector_store.load_vector_store()
    
    embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    retriever = Retriever(embedder=embedder, vector_store=vector_store)
    
    generator = Generator(use_local=True, model_name="Qwen/Qwen1.5-0.5B")
    backend_ready = True
    print("[✓] RAG Backend initialized successfully.")
except Exception as e:
    print(f"[!] Failed to initialize backend: {e}")
    backend_ready = False

def chat_interface(message, history):
    """Processes the user's message through the RAG pipeline."""
    if not backend_ready:
        return "System offline. Please ensure the knowledge base is built by running `python main.py --build-kb`."
        
    try:
        # Retrieve context
        contexts = retriever.retrieve_context(message, top_k=3)
        if not contexts:
            return "I could not find any relevant context in the database to answer your question."
            
        # Generate answer
        answer = generator.generate_answer(message, contexts)
        
        # Append sources to the answer
        source_text = "\n\n**Sources Used:**\n"
        for i, ctx in enumerate(contexts, 1):
            meta = ctx.get('metadata', {})
            source = meta.get('filename', 'Unknown')
            page = meta.get('page', 'N/A')
            distance = ctx.get('distance', 0.0)
            source_text += f"- {source} (Page {page}) [Distance: {distance:.4f}]\n"
            
        return answer + source_text
        
    except Exception as e:
        return f"An error occurred while generating the answer: {str(e)}"

# Define the Gradio Chat Interface
demo = gr.ChatInterface(
    fn=chat_interface,
    title="RAG-based Question Answering System",
    description="Ask questions based on the laboratory knowledge base. The system retrieves relevant chunks and uses Qwen1.5-0.5B to synthesize an answer.",
    examples=["What is the main advantage of the Transformer architecture?", "What does RAG combine?"],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
