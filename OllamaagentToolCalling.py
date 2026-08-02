"""
AI Agent with Tools — runs 100% locally using Ollama.
Works identically on Mac and Windows.

The agent can:
  1. Chat with memory (remembers the conversation)
  2. Decide on its own when to use tools:
     - calculator: does math
     - get_current_time: tells date/time
     - save_note / read_notes: writes & reads a notes file

Setup:
  ollama pull llama3.2          (or qwen2.5:7b for better tool use)
  python -m pip install openai
  python ollama_agent.py
"""

import json
import math
from datetime import datetime
from pathlib import Path
from openai import OpenAI

# ---------------------------------------------------------------
# 1. Connect to local Ollama (OpenAI-compatible endpoint)
# ---------------------------------------------------------------
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "llama3.2"  # change to "qwen2.5:7b" if you pulled it

NOTES_FILE = Path("agent_notes.txt")

SYSTEM_PROMPT = (
    "You are a helpful personal assistant agent. "
    "Use your tools whenever they can help: calculator for any math, "
    "get_current_time for date/time questions, save_note to remember "
    "things the user tells you, and read_notes to recall them. "
    "Answer concisely."
)

# ---------------------------------------------------------------
# 2. Tool implementations (plain Python functions)
# ---------------------------------------------------------------
def calculator(expression: str) -> str:
    """Safely evaluate a math expression like '2*(3+4)' or 'sqrt(16)'."""
    allowed = {n: getattr(math, n) for n in dir(math) if not n.startswith("_")}
    allowed.update({"abs": abs, "round": round, "min": min, "max": max})
    try:
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"Error: {e}"


def get_current_time() -> str:
    """Return the current local date and time."""
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


def save_note(note: str) -> str:
    """Append a note to the notes file."""
    with NOTES_FILE.open("a", encoding="utf-8") as f:
        f.write(note.strip() + "\n")
    return "Note saved."


def read_notes() -> str:
    """Read all saved notes."""
    if not NOTES_FILE.exists():
        return "No notes saved yet."
    return NOTES_FILE.read_text(encoding="utf-8") or "No notes saved yet."


# Map tool names -> functions
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "save_note": save_note,
    "read_notes": read_notes,
}

# ---------------------------------------------------------------
# 3. Tool schemas (what the model sees)
# ---------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression, e.g. '15/3 + sqrt(49)'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a note/reminder the user wants remembered.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "The note text"}
                },
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_notes",
            "description": "Read back all previously saved notes.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# ---------------------------------------------------------------
# 4. The agent loop
# ---------------------------------------------------------------
def run_agent():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Local AI Agent ready (model: %s). Type 'exit' to quit.\n" % MODEL)

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Agent: Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        # Inner loop: keep going while the model wants to call tools
        while True:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                # Record the assistant's tool request in history
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
                # Execute each tool and feed results back
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments or "{}")
                    print(f"  [tool call: {name}({args})]")
                    func = TOOL_FUNCTIONS.get(name)
                    result = func(**args) if func else f"Unknown tool: {name}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(result),
                        }
                    )
                continue  # let the model see the results and continue

            # Plain text answer — show it and return to the user
            print(f"Agent: {msg.content}\n")
            messages.append({"role": "assistant", "content": msg.content})
            break


if __name__ == "__main__":
    run_agent()