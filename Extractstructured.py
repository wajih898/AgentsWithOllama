"""
Example 2: STRUCTURED DATA EXTRACTION — LLM as a data parser.

Turns messy, unstructured text (emails, invoices, resumes, complaints)
into clean, validated JSON your code can actually use — insert into a
database, send to an API, display in a UI, etc.

This uses Ollama's structured output feature: you pass a JSON schema
and the model is FORCED to reply in that exact shape.

This is one of the most practical business uses of LLMs — e.g. reading
customer emails and auto-creating support tickets.

Setup:
  ollama pull llama3.2
  python -m pip install ollama pydantic
  python extract_structured.py
"""

import json
import ollama
from pydantic import BaseModel
from typing import Optional

MODEL = "llama3.2"


# ---------------------------------------------------------------
# 1. Define the structure you want (Pydantic model = schema + validation)
# ---------------------------------------------------------------
class SupportTicket(BaseModel):
    customer_name: Optional[str]
    product: Optional[str]
    issue_summary: str
    category: str            # billing / technical / delivery / other
    urgency: str             # low / medium / high
    order_id: Optional[str]
    requested_action: Optional[str]


# ---------------------------------------------------------------
# 2. Extraction function
# ---------------------------------------------------------------
def extract_ticket(raw_text: str) -> SupportTicket:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract a support ticket from the customer's message. "
                    "Infer category and urgency sensibly. Use null for "
                    "anything not mentioned."
                ),
            },
            {"role": "user", "content": raw_text},
        ],
        format=SupportTicket.model_json_schema(),  # <-- forces JSON shape
    )
    return SupportTicket.model_validate_json(response["message"]["content"])


# ---------------------------------------------------------------
# 3. Demo with messy real-world style messages
# ---------------------------------------------------------------
SAMPLE_MESSAGES = [
    """Salam, this is Ahmed Raza. I bought your SmartPOS Pro plan last month,
    order #SP-4491, and since yesterday the terminal is not syncing invoices
    at all!! We have a shop full of customers and everything is offline.
    Please fix this TODAY or refund my money.""",

    """Hi, just a quick question - I was charged $79 twice this month for my
    subscription. Can someone check? No rush, but would like it sorted before
    next billing cycle. Thanks, Sara.""",

    """Package never arrived. Tracking says delivered but nothing here.
    Order XK-2210.""",
]

if __name__ == "__main__":
    for i, msg in enumerate(SAMPLE_MESSAGES, 1):
        print(f"\n{'=' * 60}\nRAW MESSAGE {i}:\n{msg.strip()[:100]}...")
        ticket = extract_ticket(msg)
        print(f"\nEXTRACTED TICKET (valid JSON):")
        print(json.dumps(ticket.model_dump(), indent=2))

    # Try your own
    print(f"\n{'=' * 60}")
    custom = input("\nPaste any customer message to extract (or Enter to skip): ").strip()
    if custom:
        print(json.dumps(extract_ticket(custom).model_dump(), indent=2))