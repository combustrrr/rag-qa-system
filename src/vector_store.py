import chromadb
from typing import List, Dict, Any, Optional
import os

from src.embedder import EmbeddedChunk

class VectorStore:
    def __init__(self, persist_directory: str, collection_name: str = "rag_collection"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None

    def create_vector_store(self):
        """Creates or loads a persistent vector store."""
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        return self.collection

    def load_vector_store(self):
        """Loads an existing vector store."""
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_collection(name=self.collection_name)
        return self.collection

    def save_vector_store(self):
        """Saves the vector store to disk. 
        For ChromaDB PersistentClient, changes are automatically saved to SQLite/Parquet, 
        so this is mainly a placeholder to fulfill the architectural requirement.
        """
        pass

    def add_documents(self, embedded_chunks: List[EmbeddedChunk]):
        """Adds document chunks with their embeddings to the vector store."""
        if not self.collection:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")
            
        if not embedded_chunks:
            return

        ids = [chunk.chunk_id for chunk in embedded_chunks]
        texts = [chunk.text for chunk in embedded_chunks]
        metadatas = [chunk.metadata for chunk in embedded_chunks]
        embeddings = [chunk.embedding for chunk in embedded_chunks]

        # ChromaDB adds in batches, let's just add them all directly for now
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def search(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """Searches the database using a query embedding."""
        if not self.collection:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results


def run_vector_store_demo():
    """Demonstrates creating a vector database and indexing chunks."""
    from src.embedder import run_embedding_generation
    from pathlib import Path
    import shutil

    # Run the previous stages to get embedded chunks
    embedded_chunks, mapping = run_embedding_generation()
    if not embedded_chunks:
        print("[!] No chunks available to index.")
        return

    # Setup database directory
    base_dir = Path(__file__).resolve().parent.parent
    db_dir = base_dir / "data" / "chroma_db"
    
    # Clean previous demo DB if it exists
    if db_dir.exists():
        shutil.rmtree(db_dir)
        
    print("=" * 80)
    print("   STAGE 6: VECTOR DATABASE INDEXING (ChromaDB)")
    print("=" * 80)
    
    # Initialize the VectorStore
    db = VectorStore(persist_directory=str(db_dir))
    
    # 1. Create Vector Store
    print(f"[*] Creating Vector Store at: {db_dir}")
    collection = db.create_vector_store()
    
    # 2. Add Documents
    print(f"[*] Adding {len(embedded_chunks)} chunks to the database...")
    db.add_documents(embedded_chunks)
    
    # 3. Save (No-op in Chroma, but we show it)
    print("[*] Saving Vector Store...")
    db.save_vector_store()
    
    # 4. Demonstrate Loading and Search
    print("\n[*] Demonstrating Vector Database loading and searching...")
    # Instantiate a new object to prove it loads from disk
    db_loaded = VectorStore(persist_directory=str(db_dir))
    db_loaded.load_vector_store()
    
    # Use the first chunk's embedding as a fake query to see if it finds itself
    query_chunk = embedded_chunks[0]
    print(f"\n[+] Searching for chunks similar to Chunk ID: {query_chunk.chunk_id}")
    
    # 5. Search
    results = db_loaded.search(query_embedding=query_chunk.embedding, n_results=3)
    
    print("\n[+] Search Results:")
    for i in range(len(results['ids'][0])):
        match_id = results['ids'][0][i]
        distance = results['distances'][0][i]
        text_snippet = results['documents'][0][i][:60].replace("\n", " ")
        print(f"  {i+1}. ID: {match_id} | Distance: {distance:.4f}")
        print(f"     Text: \"{text_snippet}...\"")
    
    print("\n[✓] Stage 6 complete! Vector database successfully created and verified.\n")

if __name__ == "__main__":
    run_vector_store_demo()
