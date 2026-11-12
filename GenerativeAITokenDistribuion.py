"""
Module 1 - Generative AI Foundations: working examples
Python & Generative AI Bootcamp - Day 5

Setup (once):
    pip install openai python-dotenv pydantic tiktoken

Create a file named .env next to this script:
    OPENAI_API_KEY=sk-...
    GEMINI_API_KEY=...        # optional, free key from Google AI Studio
    # For local models: install Ollama and run `ollama pull llama3.2`

Run one example:
    python module1_examples.py 1      (or 2 ... 9)
Run all:
    python module1_examples.py all
"""

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()

# Model IDs change over time - check your provider's models page if one stops working.
OPENAI_MODEL = "gpt-6-luna"          # OpenAI's efficient, low-cost model
GEMINI_MODEL = "gemini-3.6-flash"    # Google Gemini (stable Flash model)
OLLAMA_MODEL = "llama3.2"            # local open-source model via Ollama


def openai_client() -> OpenAI:
    return OpenAI()  # reads OPENAI_API_KEY from the environment


def ask(prompt: str, system: str | None = None, temperature: float = 0.3,
        max_tokens: int = 400, client: OpenAI | None = None, model: str = OPENAI_MODEL) -> str:
    """Small helper: send one prompt (plus optional system message) and return the text."""
    client = client or openai_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Example 1: Tokens - how an LLM "sees" text (runs offline after first download)
# ---------------------------------------------------------------------------
def ex1_tokens():
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    for text in ["Generative AI is transforming Pakistan's tech industry.",
                 "مصنوعی ذہانت پاکستان کی ٹیک انڈسٹری کو بدل رہی ہے۔"]:
        ids = enc.encode(text)
        pieces = [enc.decode([i]) for i in ids]
        print(f"\nText: {text}")
        print(f"Characters: {len(text)}  |  Tokens: {len(ids)}")
        print("Pieces:", pieces)
    print("\nTakeaway: you pay per token, and non-English text often needs more tokens.")


# ---------------------------------------------------------------------------
# Example 2: Your first LLM call
# ---------------------------------------------------------------------------
def ex2_first_call():
    client = openai_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": "Explain Generative AI in 2 sentences."}],
        temperature=0.3,
        max_tokens=150,
    )
    print(response.choices[0].message.content)
    print("\nTokens used:", response.usage.total_tokens)


# ---------------------------------------------------------------------------
# Example 3: Temperature - same prompt, different creativity
# ---------------------------------------------------------------------------
def ex3_temperature():
    prompt = "Write a one-line tagline for a chai cafe in Islamabad."
    for temp in (0.0, 1.0):
        print(f"\n--- temperature = {temp} ---")
        for _ in range(3):
            print("*", ask(prompt, temperature=temp, max_tokens=40))


# ---------------------------------------------------------------------------
# Example 4: One prompt, three providers (OpenAI-compatible APIs)
# ---------------------------------------------------------------------------
def ex4_providers():
    providers = {
        "OpenAI": (OpenAI(), OPENAI_MODEL),
        "Gemini": (OpenAI(api_key=os.getenv("GEMINI_API_KEY"),
                          base_url="https://generativelanguage.googleapis.com/v1beta/openai/"),
                   GEMINI_MODEL),
        "Ollama (local)": (OpenAI(api_key="ollama", base_url="http://localhost:11434/v1"),
                           OLLAMA_MODEL),
    }
    prompt = "In one sentence, what is a Large Language Model?"
    for name, (client, model) in providers.items():
        try:
            print(f"\n[{name} - {model}]\n", ask(prompt, client=client, model=model, max_tokens=80))
        except Exception as e:  # missing key or Ollama not running
            print(f"\n[{name}] skipped: {e.__class__.__name__}: {e}")


# ---------------------------------------------------------------------------
# Example 5: Zero-shot vs few-shot
# ---------------------------------------------------------------------------
def ex5_few_shot():
    zero = "Classify the sentiment of this review as Positive, Negative or Mixed:\n" \
           "'Delivery was late but the kurta quality is excellent.'"
    few = """Classify the sentiment as Positive, Negative or Mixed. Reply with one word.

Review: "Fabric is soft and fits perfectly." -> Positive
Review: "Colour faded after one wash." -> Negative
Review: "Nice design, but sizes run small." -> Mixed
Review: "Delivery was late but the kurta quality is excellent." ->"""
    print("Zero-shot:\n", ask(zero, temperature=0))
    print("\nFew-shot:\n", ask(few, temperature=0))


# ---------------------------------------------------------------------------
# Example 6: Chain-of-thought
# ---------------------------------------------------------------------------
def ex6_chain_of_thought():
    q = ("A shirt costs Rs 2,500. The shop gives 20% off, then adds 5% sales tax "
         "on the discounted price. What is the final price?")
    print("Direct:\n", ask(q + " Give only the number.", temperature=0))
    print("\nStep by step:\n", ask(q + " Think step by step, then give the final answer.",
                                   temperature=0))
    # Correct: 2500 x 0.80 = 2000; 2000 x 1.05 = Rs 2,100


# ---------------------------------------------------------------------------
# Example 7: System vs user messages
# ---------------------------------------------------------------------------
def ex7_system_message():
    system = ("You are a friendly Python tutor for beginners at a Pakistani IT institute. "
              "Explain with simple analogies, keep answers under 80 words, "
              "and end with one practice question.")
    print(ask("What is a Python dictionary?", system=system))


# ---------------------------------------------------------------------------
# Example 8: Structured JSON output, validated with Pydantic
# ---------------------------------------------------------------------------
class ReviewAnalysis(BaseModel):
    sentiment: str          # Positive | Negative | Mixed
    product: str
    issues: list[str]
    rating_guess: int       # 1-5


def analyze_review(review: str) -> ReviewAnalysis:
    client = openai_client()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content":
                "Extract review data. Return ONLY JSON with keys: "
                "sentiment (Positive|Negative|Mixed), product (string), "
                "issues (list of strings), rating_guess (integer 1-5)."},
            {"role": "user", "content": review},
        ],
    )
    data = json.loads(resp.choices[0].message.content)
    return ReviewAnalysis(**data)   # raises ValidationError if the shape is wrong


def ex8_json_output():
    review = "The lawn suit colours are beautiful but stitching came loose after a week."
    try:
        result = analyze_review(review)
        print(result.model_dump_json(indent=2))
    except ValidationError as e:
        print("Model returned unexpected JSON:", e)


# ---------------------------------------------------------------------------
# Example 9: LAB - batch of customer reviews -> JSON file
# ---------------------------------------------------------------------------
REVIEWS = [
    "Shoes are comfortable and arrived in 2 days. Very happy!",
    "Wrong size delivered and customer service never replied.",
    "Good value for money, though the box was damaged.",
    "Watch strap broke within a month. Not buying again.",
]


def ex9_lab():
    results = []
    for r in REVIEWS:
        a = analyze_review(r)
        results.append({"review": r, **a.model_dump()})
        print(f"{a.sentiment:<9} {a.rating_guess}/5  {a.product}")
    with open("reviews_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSaved reviews_analysis.json")


EXAMPLES = {
    "1": ex1_tokens, "2": ex2_first_call, "3": ex3_temperature,
    "4": ex4_providers, "5": ex5_few_shot, "6": ex6_chain_of_thought,
    "7": ex7_system_message, "8": ex8_json_output, "9": ex9_lab,
}

if __name__ == "__main__":
    choice = sys.argv[1] if len(sys.argv) > 1 else "1"
    targets = EXAMPLES.values() if choice == "all" else [EXAMPLES[choice]]
    for fn in targets:
        print(f"\n===== {fn.__name__} =====")
        fn()