# Multi-Agent Collaboration with Ollama

This project demonstrates a simple multi-agent workflow using Ollama and Python. It shows how two agents can work together to improve a piece of content through collaboration.

## What this example shows

The script uses two agents with different roles:

- A Writer creates an initial draft
- A Critic reviews the draft and asks for improvements
- The process repeats until the Critic approves the result

This pattern is similar to the style used by frameworks such as CrewAI and AutoGen, where multiple LLM-based agents interact with each other through different prompts and responsibilities.

## How it works

1. The Writer generates a first version of the content.
2. The Critic evaluates it and gives feedback.
3. The Writer updates the content based on that feedback.
4. The process continues until the Critic says "APPROVED".

This creates a loop of refinement, making the final output stronger and more polished.

## Why this is useful

Multi-agent systems are useful for tasks like:

- Writing and editing content
- Reviewing code or documents
- Comparing ideas from different perspectives
- Building more realistic AI workflows

## Prerequisites

Before running the example, make sure you have:

- Ollama installed and running
- A local model downloaded, such as `llama3.2`
- Python installed on your system

## Setup

Install the required Python package:

```bash
python3 -m pip install openai
```

Pull a model if needed:

```bash
ollama pull llama3.2
```

## Run the example

Start Ollama if it is not already running:

```bash
ollama serve
```

Then run the script:

```bash
python3 multi_agent.py
```

## Expected behavior

You should see the draft improve over several rounds as the Critic gives feedback and the Writer revises the content.

## Summary

This example is a simple introduction to multi-agent AI systems. It shows how multiple agents can collaborate to produce better results than a single model acting alone.
