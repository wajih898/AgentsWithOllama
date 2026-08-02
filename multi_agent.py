"""
Example 1: MULTI-AGENT SYSTEM — two agents collaborating.

A "Writer" agent produces a draft, a "Critic" agent reviews it,
and the Writer improves the draft based on the feedback. This
loop repeats until the Critic approves (or max rounds hit).

This is the core pattern behind CrewAI / AutoGen style frameworks:
multiple LLM calls with different system prompts (roles) passing
work between each other.

Setup:
  ollama pull llama3.2
  python -m pip install ollama
  python multi_agent.py
"""

import ollama

MODEL = "llama3.2"
MAX_ROUNDS = 3

WRITER_PROMPT = (
    "You are a professional writer. Write clear, engaging content. "
    "If you receive feedback, revise your previous draft to address "
    "every point. Output ONLY the draft text, nothing else."
)

CRITIC_PROMPT = (
    "You are a strict editor. Review the draft you are given. "
    "If it is genuinely good, reply with exactly: APPROVED. "
    "Otherwise list 2-3 specific, actionable improvements as bullet points. "
    "Be demanding about clarity, structure, and engagement."
)


def call_agent(system_prompt: str, user_message: str) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response["message"]["content"].strip()


def run_pipeline(task: str):
    print(f"\nTASK: {task}\n{'=' * 60}")

    draft = call_agent(WRITER_PROMPT, task)
    print(f"\n--- WRITER (draft 1) ---\n{draft}")

    for round_num in range(1, MAX_ROUNDS + 1):
        feedback = call_agent(CRITIC_PROMPT, f"Review this draft:\n\n{draft}")
        print(f"\n--- CRITIC (round {round_num}) ---\n{feedback}")

        if "APPROVED" in feedback.upper():
            print("\nCritic approved the draft!")
            break

        draft = call_agent(
            WRITER_PROMPT,
            f"Task: {task}\n\nYour previous draft:\n{draft}\n\n"
            f"Editor feedback:\n{feedback}\n\nRevise the draft.",
        )
        print(f"\n--- WRITER (revised draft {round_num + 1}) ---\n{draft}")

    print(f"\n{'=' * 60}\nFINAL OUTPUT:\n{draft}")


if __name__ == "__main__":
    task = input("What should the agents write? (e.g. 'a LinkedIn post about AI agents'): ").strip()
    if not task:
        task = "a short LinkedIn post announcing our new AI chatbot product"
    run_pipeline(task)