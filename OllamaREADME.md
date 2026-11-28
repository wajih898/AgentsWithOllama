# Ollama Agent

This project runs a small local AI agent using Ollama and Python. The agent can chat with you, remember the conversation, and decide when to use tools such as a calculator, a clock, and a notes file.

## What this script does

The file [Ollamaagent.py](Ollamaagent.py) does the following:

1. Connects to a locally running Ollama server using the OpenAI-compatible API.
2. Sends your messages to a local model such as `llama3.2`.
3. Lets the model decide whether it should call a tool.
4. Executes the tool locally in Python.
5. Sends the tool result back to the model so it can continue responding.

## Supported tools

The agent can use these tools:

- `calculator`: evaluates simple math expressions
- `get_current_time`: returns the current date and time
- `save_note`: saves a note to a file
- `read_notes`: reads saved notes back

## Prerequisites

Install Ollama first:

- macOS / Windows: https://ollama.com/

python -m venv venv
source venv/bin/activate #Mac Users
venv\Scripts\activate # windows
pip install -r requirements.txt


After installing Ollama, pull a model:

```bash
ollama pull llama3.2
```

You can also use `qwen2.5:7b` if you prefer.

## Install Python dependency

```bash
python3 -m pip install openai
```

## Run the agent

Start the Ollama server:

```bash
ollama serve
```

In a second terminal, run:

```bash
python3 Ollamaagent.py
```

## How to use it

Once the script is running, type messages like:

```text
What time is it?
```

```text
What is 25 * 4?
```

```text
Remember that I need to submit my assignment tomorrow.
```

Type `exit`, `quit`, or `bye` to stop the program.

## What is happening behind the scenes

The script creates a chat loop:

- Your input is added to the conversation history.
- The model is asked to answer.
- If the model decides a tool is helpful, it requests that tool.
- The Python function runs the tool.
- The result is sent back to the model.
- The model then produces the final response.

This is the basic pattern for an agent with tool use.

## Notes file

Saved notes are stored in a file named `agent_notes.txt` in the same folder as the script.


## Commads Need to know
# Start the server (skip this if you installed the desktop app — it auto-starts)
ollama serve

# Download a model
ollama pull llama3.2

# Chat with it in the terminal (Ctrl+D or /bye to exit)
ollama run llama3.2

# List downloaded models
ollama list

# Check it's running / see loaded models
ollama ps

# Remove a model
ollama rm llama3.2

## Python side (also identical on both):
python -m pip install openai