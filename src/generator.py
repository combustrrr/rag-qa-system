import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Try to load env variables for optional API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class Generator:
    """Handles the generation stage of the RAG pipeline."""

    def __init__(self, use_local: bool = True, model_name: str = "Qwen/Qwen1.5-0.5B"):
        """Initializes the generator model.
        
        Args:
            use_local: If True, uses a local HuggingFace transformers model.
                       If False, attempts to use an external API (requires env vars).
            model_name: The name of the local HuggingFace model to load.
        """
        self.use_local = use_local
        self.model_name = model_name
        self._pipeline = None
        self._api_key = None

        if self.use_local:
            logger.info(f"Loading local LLM '{self.model_name}'...")
            try:
                from transformers import pipeline
                self._pipeline = pipeline("text-generation", model=self.model_name, device="cpu")
            except ImportError as exc:
                raise ImportError(
                    "transformers and torch are required for local LLM generation. "
                    "Install them using: pip install transformers torch"
                ) from exc
        else:
            self._api_key = os.environ.get("OPENAI_API_KEY")
            if not self._api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not found. "
                    "Please set it or use use_local=True for free local generation."
                )
            logger.info("Using API-based LLM generation.")

    def construct_prompt(self, question: str, retrieved_context: List[Dict[str, Any]]) -> str:
        """Constructs the prompt combining context and question based on instructions."""
        
        # Combine retrieved chunks into a single context string
        context_parts = []
        for i, ctx in enumerate(retrieved_context, 1):
            source = ctx.get('metadata', {}).get('filename', 'Unknown source')
            page = ctx.get('metadata', {}).get('page', 'Unknown page')
            text = ctx.get('text', '')
            context_parts.append(f"--- Context {i} (Source: {source}, Page: {page}) ---\n{text}")
            
        combined_context = "\n\n".join(context_parts)
        
        # Build the final prompt per the guidelines
        prompt = (
            "You are a helpful and precise assistant. Answer the user's question using ONLY the supplied context.\n"
            "Follow these rules exactly:\n"
            "1. Do not invent or fabricate information (no hallucination).\n"
            "2. If the answer is not present in the context, clearly say: 'I could not find the answer in the provided documents.'\n"
            "3. Provide a concise answer.\n"
            "4. Mention the source document and page number in your answer if applicable.\n\n"
            "SUPPLIED CONTEXT:\n"
            f"{combined_context}\n\n"
            "QUESTION:\n"
            f"{question}\n\n"
            "ANSWER:"
        )
        return prompt

    def generate_answer(self, question: str, retrieved_context: List[Dict[str, Any]]) -> str:
        """Generates an answer to the question based on the retrieved context."""
        
        # 1. Construct the prompt
        prompt = self.construct_prompt(question, retrieved_context)
        
        # 2. Output the complete prompt sent to the LLM (for demonstration/debugging)
        print("\n" + "=" * 80)
        print("   PROMPT SENT TO LLM")
        print("=" * 80)
        print(prompt)
        print("=" * 80 + "\n")
        
        # 3. Generate answer
        if self.use_local:
            result = self._pipeline(prompt, max_new_tokens=150, num_return_sequences=1)
            # Strip the prompt from the output
            generated_text = result[0]['generated_text']
            return generated_text[len(prompt):].strip()
        else:
            # Fallback to a basic OpenAI implementation if API is used
            import requests
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}"
            }
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0, # low temperature for factual RAG
                "max_tokens": 150
            }
            response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()


# Standalone function as requested
_GLOBAL_GENERATOR = None

def generate_answer(question: str, retrieved_context: List[Dict[str, Any]], use_local: bool = True) -> str:
    """Standalone convenience function to generate an answer."""
    global _GLOBAL_GENERATOR
    if _GLOBAL_GENERATOR is None or _GLOBAL_GENERATOR.use_local != use_local:
        # Use a very small model for testing if local
        _GLOBAL_GENERATOR = Generator(use_local=use_local, model_name="Qwen/Qwen1.5-0.5B")
        
    return _GLOBAL_GENERATOR.generate_answer(question, retrieved_context)

def run_generation_demo():
    """Demonstrates combining retrieval and generation."""
    from src.retriever import retrieve_context
    
    print("=" * 80)
    print("   STAGE 8: GENERATION PIPELINE")
    print("=" * 80)
    
    question = "What is the main advantage of the Transformer architecture?"
    print(f"[*] User Question: {question}")
    print("[*] Retrieving context from Vector Database...")
    
    # Retrieve top 3 chunks
    contexts = retrieve_context(question, top_k=3)
    
    if not contexts:
        print("[!] No contexts retrieved. Have you run the previous stages?")
        return
        
    print(f"[*] Retrieved {len(contexts)} chunks. Initializing Generator...")
    
    # Generate answer
    print("[*] Calling LLM (this may take a moment to load weights/generate)...")
    answer = generate_answer(question, contexts, use_local=True)
    
    print("\n" + "=" * 80)
    print("   FINAL GENERATED ANSWER")
    print("=" * 80)
    print(answer)
    print("=" * 80 + "\n")
    print("[✓] Stage 8 complete! Generation pipeline verified.\n")


if __name__ == "__main__":
    # Ensure warnings are minimized during lab
    import warnings
    warnings.filterwarnings("ignore")
    run_generation_demo()
