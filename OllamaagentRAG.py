"""
RAG (Retrieval-Augmented Generation) — 100% local with Ollama.
Works identically on Mac and Windows.

Pipeline:
  1. Load documents (from the docs/ folder, or built-in samples)
  2. Split them into chunks
  3. Embed each chunk with Ollama's embedding model  -> vectors
  4. On each question: embed the question, find the most
     similar chunks (cosine similarity), and give them to the
     LLM as context to answer from.

Setup:
  ollama pull llama3.2
  ollama pull nomic-embed-text
  python -m pip install ollama numpy
  python rag_ollama.py
"""

from pathlib import Path
import numpy as np
import ollama

CHAT_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"
DOCS_FOLDER = Path("docs")   # put your .txt / .md files here (optional)
CHUNK_SIZE = 500             # characters per chunk
TOP_K = 3                    # how many chunks to retrieve per question

# Sample knowledge base used if no docs/ folder exists.
SAMPLE_DOCS = {
    "company_policy.txt": """
        Systems Limited leave policy: Employees get 20 annual leaves per year.
        Sick leaves are 10 per year and require a medical certificate if more
        than 2 consecutive days. Casual leaves must be applied 3 days in advance.
        Unused annual leaves can carry over, maximum 10 days into the next year.
    """,
    "product_faq.txt": """
        Our product SmartPOS supports offline billing. When the internet is
        restored, all invoices sync automatically to the cloud dashboard.
        SmartPOS pricing: Basic plan is $29/month for 1 terminal, Pro plan
        is $79/month for up to 5 terminals with inventory management.
        Refunds are processed within 5-7 business days.
    """,
    "it_support.txt": """
        To reset your office VPN password, visit portal.company.com/vpn and
        click Forgot Password. VPN sessions expire after 12 hours. For laptop
        issues, raise a ticket at helpdesk.company.com; hardware replacement
        takes 3 working days after approval by your line manager.
    """,
}

# ---------------------------------------------------------------
# 1. Load documents
# ---------------------------------------------------------------
def load_documents() -> dict:
    docs = {}
    if DOCS_FOLDER.exists():
        for f in DOCS_FOLDER.glob("*"):
            if f.suffix.lower() in (".txt", ".md"):
                docs[f.name] = f.read_text(encoding="utf-8", errors="ignore")
    if not docs:
        print("(no docs/ folder found — using built-in sample documents)\n")
        docs = SAMPLE_DOCS
    return docs


# ---------------------------------------------------------------
# 2. Chunking — split text into overlapping pieces
# ---------------------------------------------------------------
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = 50) -> list:
    text = " ".join(text.split())  # normalize whitespace
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


# ---------------------------------------------------------------
# 3. Embeddings + vector index (in memory)
# ---------------------------------------------------------------
def embed(text: str) -> np.ndarray:
    result = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return np.array(result["embedding"])


def build_index(docs: dict):
    """Return a list of (chunk_text, source_name, vector)."""
    index = []
    for name, content in docs.items():
        for chunk in chunk_text(content):
            index.append((chunk, name, embed(chunk)))
            print(f"  embedded chunk from {name} ({len(chunk)} chars)")
    return index


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(question: str, index, top_k: int = TOP_K):
    """Find the top_k most relevant chunks for the question."""
    q_vec = embed(question)
    scored = [
        (cosine_similarity(q_vec, vec), chunk, source)
        for chunk, source, vec in index
    ]
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:top_k]


# ---------------------------------------------------------------
# 4. Generation — answer using ONLY the retrieved context
# ---------------------------------------------------------------
def answer(question: str, index) -> str:
    results = retrieve(question, index)

    context = "\n\n".join(
        f"[Source: {source} | relevance: {score:.2f}]\n{chunk}"
        for score, chunk, source in results
    )

    print("\n--- retrieved context ---")
    for score, chunk, source in results:
        print(f"  {score:.2f}  {source}: {chunk[:60]}...")
    print("-------------------------\n")

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on my documents."
Mention which source you used.

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("Loading and embedding documents (one-time step)...")
    docs = load_documents()
    index = build_index(docs)
    print(f"\nRAG ready! {len(index)} chunks indexed. Ask questions, 'exit' to quit.\n")

    while True:
        q = input("Question: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit", "bye"):
            break
        print(f"\nAnswer: {answer(q, index)}\n")