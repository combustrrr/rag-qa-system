from typing import List, Dict, Any
import logging

from src.embedder import EmbeddingGenerator, DEFAULT_MODEL_NAME
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

class Retriever:
    """Handles the retrieval stage of the RAG pipeline."""

    def __init__(self, embedder: EmbeddingGenerator, vector_store: VectorStore):
        """Initializes the retriever with an embedder and a vector store.
        
        Args:
            embedder: Generator used to encode the query into the exact same vector space as the documents.
            vector_store: The ChromaDB VectorStore instance containing the document chunks.
        """
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve_context(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves the most semantically similar chunks for a given question.
        
        Args:
            question: The user's query string.
            top_k: Number of most similar chunks to retrieve.
            
        Returns:
            List of dictionaries containing retrieved text, metadata, and distance.
        """
        # 1. Convert the question into an embedding using the SAME model
        query_embedding = self.embedder.embed_text(question)
        
        # 2. Search the vector database using the generated embedding
        results = self.vector_store.search(query_embedding=query_embedding, n_results=top_k)
        
        # 3. Process and format the results
        retrieved_contexts = []
        
        if not results.get('ids') or not results['ids'][0]:
            return retrieved_contexts
            
        for i in range(len(results['ids'][0])):
            context = {
                "chunk_id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if results.get('distances') else 0.0
            }
            retrieved_contexts.append(context)
            
        return retrieved_contexts


def print_embedding_space_theory():
    """Explains why the query must use the exact same embedding space as the documents."""
    print("=" * 80)
    print("   THEORETICAL INSIGHT: THE SHARED EMBEDDING SPACE")
    print("=" * 80)
    print("Why must the question use the exact same embedding model as the documents?")
    print("\n1. Shared Semantic Vector Space:")
    print("   Think of an embedding model as a cartographer mapping out meaning in a city.")
    print("   If the documents are mapped using one coordinate system (Model A), and the")
    print("   question is mapped using a completely different system (Model B), their")
    print("   coordinates (vectors) will be in entirely different spatial dimensions.")
    print("\n2. Meaningful Distance:")
    print("   Similarity algorithms like Cosine Distance or L2 Norm only work if the points")
    print("   exist in the exact same mathematical space. By using the same model, we ensure")
    print("   that a question like 'What is attention?' is placed geometrically close to")
    print("   document chunks explaining the Transformer attention mechanism.")
    print("=" * 80 + "\n")


def run_retrieval_demo():
    """Demonstrates loading the DB, embedding a query, and retrieving context."""
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent.parent
    db_dir = base_dir / "data" / "chroma_db"
    
    if not db_dir.exists():
        print("[!] Vector database not found. Please run src.vector_store first.")
        return

    print("=" * 80)
    print("   STAGE 7: SEMANTIC RETRIEVAL PIPELINE")
    print("=" * 80)
    
    # Initialize the Embedder (Must be the SAME model used in Stage 5)
    print("[*] Loading embedding model (for queries)...")
    embedder = EmbeddingGenerator(model_name=DEFAULT_MODEL_NAME)
    
    # Initialize and load the Vector Store
    print(f"[*] Loading Vector Store from: {db_dir}")
    vector_store = VectorStore(persist_directory=str(db_dir))
    vector_store.load_vector_store()
    
    # Initialize the Retriever
    retriever = Retriever(embedder=embedder, vector_store=vector_store)
    
    # Example queries to test
    questions = [
        "What is the main advantage of the Transformer architecture?",
        "Do recurrent neural networks take long to train?",
    ]
    
    for q_idx, question in enumerate(questions, 1):
        print(f"\n[QUERY {q_idx}] {question}")
        
        # Retrieve context
        top_k = 3
        results = retriever.retrieve_context(question=question, top_k=top_k)
        
        print(f"[*] Retrieved {len(results)} chunks:")
        print("-" * 80)
        
        for i, res in enumerate(results, 1):
            chunk_id = res['chunk_id']
            distance = res['distance']
            text = res['text']
            meta = res['metadata']
            
            source = meta.get('filename', 'Unknown')
            page = meta.get('page', 'N/A')
            
            # Print the required fields
            print(f"  {i}. Source Document: {source} (Page: {page})")
            print(f"     Distance Score:  {distance:.4f} (Lower = More Similar)")
            print(f"     Chunk ID:        {chunk_id}")
            print(f"     Retrieved Chunk: \"{text[:120]}...\"")
            print("-" * 80)

    print()
    print_embedding_space_theory()
    print("[✓] Stage 7 complete! Context retrieval successfully verified.\n")


if __name__ == "__main__":
    run_retrieval_demo()

# Standalone convenience function as requested
_GLOBAL_RETRIEVER = None

def retrieve_context(question: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Standalone convenience function to retrieve context using a global retriever."""
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        from pathlib import Path
        base_dir = Path(__file__).resolve().parent.parent
        db_dir = base_dir / "data" / "chroma_db"
        
        embedder = EmbeddingGenerator(model_name=DEFAULT_MODEL_NAME)
        vector_store = VectorStore(persist_directory=str(db_dir))
        vector_store.load_vector_store()
        
        _GLOBAL_RETRIEVER = Retriever(embedder=embedder, vector_store=vector_store)
        
    return _GLOBAL_RETRIEVER.retrieve_context(question, top_k)
