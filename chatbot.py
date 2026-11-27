"""
TERMINAL CHATBOT — a multi-turn chat with a local Ollama model.

Features:
  - Remembers the conversation (full message history is sent each turn)
  - Streams the reply token-by-token, like ChatGPT
  - Slash commands: /help, /clear, /history, /system, /model, /save, /exit

Setup:
  ollama pull llama3.2
  python -m pip install ollama
  python chatbot.py                 (or: python chatbot.py <model-name>)
"""

import json
import sys
from datetime import datetime

import ollama

DEFAULT_MODEL = "llama3.2"
DEFAULT_SYSTEM_PROMPT = "You are a helpful, friendly assistant. Keep answers clear and concise."

HELP_TEXT = """
Commands:
  /help            Show this help
  /clear           Forget the conversation (keeps the system prompt)
  /history         Show the conversation so far
  /system <text>   Set a new system prompt (also clears history)
  /model <name>    Switch to another Ollama model
  /save            Save the conversation to a JSON file
  /exit            Quit (also: exit, quit, bye)
"""


class ChatBot:
    def __init__(self, model: str = DEFAULT_MODEL, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.model = model
        self.system_prompt = system_prompt
        self.messages = []
        self.reset()

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def ask(self, user_input: str) -> str:
        """Send the user message with full history, stream and return the reply."""
        self.messages.append({"role": "user", "content": user_input})
        reply = ""
        try:
            for chunk in ollama.chat(model=self.model, messages=self.messages, stream=True):
                piece = chunk["message"]["content"]
                reply += piece
                print(piece, end="", flush=True)
        except Exception:
            # Don't keep a user message that never got an answer
            self.messages.pop()
            raise
        print()
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def show_history(self) -> None:
        turns = [m for m in self.messages if m["role"] != "system"]
        if not turns:
            print("(no messages yet)")
            return
        for m in turns:
            who = "You" if m["role"] == "user" else "Bot"
            print(f"{who}: {m['content']}\n")

    def save(self) -> str:
        filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"model": self.model, "messages": self.messages}, f, indent=2, ensure_ascii=False)
        return filename


def handle_command(bot: ChatBot, line: str) -> bool:
    """Run a slash command. Returns False when the app should exit."""
    cmd, _, arg = line.partition(" ")
    cmd, arg = cmd.lower(), arg.strip()

    if cmd == "/exit":
        return False
    elif cmd == "/help":
        print(HELP_TEXT)
    elif cmd == "/clear":
        bot.reset()
        print("Conversation cleared.")
    elif cmd == "/history":
        bot.show_history()
    elif cmd == "/system":
        if not arg:
            print(f"Current system prompt: {bot.system_prompt}")
        else:
            bot.system_prompt = arg
            bot.reset()
            print("System prompt updated and conversation cleared.")
    elif cmd == "/model":
        if not arg:
            print(f"Current model: {bot.model}")
        else:
            bot.model = arg
            print(f"Switched to model: {arg}")
    elif cmd == "/save":
        print(f"Saved to {bot.save()}")
    else:
        print(f"Unknown command: {cmd}. Type /help for options.")
    return True


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    bot = ChatBot(model=model)

    print("=" * 50)
    print(f"  Terminal ChatBot  (model: {bot.model})")
    print("  Type /help for commands, /exit to quit.")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Goodbye!")
            break
        if user_input.startswith("/"):
            if not handle_command(bot, user_input):
                print("Goodbye!")
                break
            continue

        print("Bot: ", end="", flush=True)
        try:
            bot.ask(user_input)
        except KeyboardInterrupt:
            print("\n(reply interrupted)")
        except ollama.ResponseError as e:
            print(f"\nModel error: {e.error}  (try: ollama pull {bot.model})")
        except Exception as e:
            print(f"\nCould not reach Ollama: {e}  (is `ollama serve` running?)")


if __name__ == "__main__":
    main()
