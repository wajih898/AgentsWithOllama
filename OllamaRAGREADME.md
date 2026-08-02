# Ollama RAG Agent

This project demonstrates a simple Retrieval-Augmented Generation (RAG) workflow using Ollama locally. It reads documents from a knowledge base, retrieves the most relevant chunks for a question, and asks a local model to answer using only those chunks.

## Requirements

Make sure you have the following installed:

- Ollama
- Python 3.9 or newer
- The following models:
  - `llama3.2`
  - `nomic-embed-text`

## Setup

Run these commands in your terminal:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
python -m pip install ollama numpy
```

If Ollama is not already running, start it with:

```bash
ollama serve
```

## Run the project

Start the agent with:

```bash
python OllamaagentRAG.py
```

## How the RAG pipeline works

The script follows four main steps:

1. **Load and chunk**  
   Documents are split into overlapping chunks so important context is not lost at boundaries.

2. **Embed and index**  
   Each chunk is converted into a vector using `nomic-embed-text`, which helps the system find semantically similar content.

3. **Retrieve**  
   When you ask a question, the question is embedded too, and the system finds the most relevant chunks using similarity search.

4. **Generate**  
   The retrieved chunks are inserted into the prompt as context, and the model answers only from that information.

## Sample questions

The project includes built-in sample documents, so you can test it right away with questions such as:

- "How many sick leaves do employees get?"
- "What's the price of the Pro plan?"
- "How do I reset my VPN password?"
- "Who is the CEO of Google?"

For questions that are not covered by the documents, the agent should respond by saying it does not know based on the provided knowledge base.

## What you will see

Before each answer, the script prints the retrieved chunks and their relevance scores so you can watch the retrieval process happen in real time.

## Using your own knowledge base

To use your own documents, create a `docs/` folder next to the script and place `.txt` or `.md` files inside it. The agent will use those files as its knowledge base.
