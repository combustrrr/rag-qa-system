import re
from typing import List, Dict, Any, Tuple
from collections import Counter
from rich.console import Console
from rich.table import Table

def normalize_text(text: str) -> str:
    """Lowercases, removes punctuation, and strips whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def compute_exact_match(prediction: str, truth: str) -> bool:
    """Calculates Exact Match (EM) after normalization."""
    return normalize_text(prediction) == normalize_text(truth)

def compute_f1(prediction: str, truth: str) -> float:
    """Calculates Token-level F1 score."""
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    if not pred_tokens or not truth_tokens:
        return 0.0
        
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate_faithfulness(prediction: str, context: str) -> float:
    """Simple explainable faithfulness: percentage of predicted words found in the context."""
    pred_tokens = normalize_text(prediction).split()
    ctx_tokens = set(normalize_text(context).split())
    
    if not pred_tokens:
        return 1.0
        
    overlap = sum(1 for token in pred_tokens if token in ctx_tokens)
    return overlap / len(pred_tokens)

def evaluate_context_relevance(truth: str, context: str) -> bool:
    """Simple context relevance: is the expected answer contained in the retrieved context?"""
    return normalize_text(truth) in normalize_text(context)

def run_evaluation():
    from src.retriever import retrieve_context
    from src.generator import generate_answer
    import warnings
    warnings.filterwarnings("ignore")
    
    # 1. Create a small evaluation dataset based ONLY on the sample_docs
    eval_dataset = [
        {
            "question": "In what year was the Transformer model introduced?",
            "expected": "2017"
        },
        {
            "question": "What is another name for self-attention?",
            "expected": "intra-attention"
        },
        {
            "question": "What does RAG combine?",
            "expected": "pre-trained parametric memory with non-parametric memory"
        },
        {
            "question": "What is the fourth step in a typical RAG pipeline?",
            "expected": "Storing vectors in a vector database such as ChromaDB"
        },
        {
            "question": "What paper introduced the Transformer model?",
            "expected": "Attention Is All You Need"
        }
    ]
    
    console = Console()
    console.print("\n[bold cyan]STAGE 10: RAG SYSTEM EVALUATION PIPELINE[/bold cyan]")
    console.print("[*] Running system over evaluation dataset. Please wait...")
    
    results = []
    
    for i, item in enumerate(eval_dataset, 1):
        q = item["question"]
        expected = item["expected"]
        
        # Run Retrieval
        contexts = retrieve_context(q, top_k=3)
        combined_context = " ".join([c["text"] for c in contexts])
        
        # Run Generation
        answer = generate_answer(q, contexts)
        
        # Calculate Metrics
        em = compute_exact_match(answer, expected)
        f1 = compute_f1(answer, expected)
        faithfulness = evaluate_faithfulness(answer, combined_context)
        ctx_relevance = evaluate_context_relevance(expected, combined_context)
        
        # Store for display
        results.append({
            "q": q,
            "expected": expected,
            "generated": answer,
            "context_len": len(combined_context),
            "em": em,
            "f1": f1,
            "faithfulness": faithfulness,
            "ctx_relevance": ctx_relevance
        })
        
    # Generate Table
    table = Table(title="RAG Evaluation Results", show_lines=True)
    table.add_column("Question", justify="left", style="cyan", max_width=20)
    table.add_column("Expected Answer", justify="left", style="green", max_width=20)
    table.add_column("Generated Answer", justify="left", style="yellow", max_width=25)
    table.add_column("EM", justify="center")
    table.add_column("F1", justify="center")
    table.add_column("Faithfulness\n(Token Overlap)", justify="center")
    table.add_column("Context\nRelevance", justify="center")
    
    for r in results:
        table.add_row(
            r["q"],
            r["expected"],
            r["generated"],
            "Yes" if r["em"] else "No",
            f"{r['f1']:.2f}",
            f"{r['faithfulness']:.0%}",
            "High" if r["ctx_relevance"] else "Low"
        )
        
    console.print(table)
    
    print("\n" + "=" * 80)
    print("   LIMITATIONS OF THIS EVALUATION (LABORATORY INSIGHT)")
    print("=" * 80)
    print("1. Token-level metrics (Exact Match, F1) are brittle: If the model outputs")
    print("   'In 2017' instead of '2017', Exact Match fails and F1 decreases, even though")
    print("   the answer is semantically correct. This penalizes chat models that are")
    print("   conversational by default.")
    print("2. Simple Faithfulness heuristic: Measuring vocabulary overlap ignores synonyms,")
    print("   morphology, and sentence structure. An LLM might rephrase the context")
    print("   correctly without using the exact tokens, scoring low on overlap.")
    print("3. LLM-as-a-Judge alternative: In modern production RAG, an advanced LLM (e.g. GPT-4)")
    print("   is typically used to evaluate semantic relevance, faithfulness, and correctness")
    print("   rather than relying purely on brittle lexical string matching.")
    print("=" * 80 + "\n")
    print("[✓] Stage 10 complete! Laboratory evaluation successfully verified.")

if __name__ == "__main__":
    run_evaluation()
