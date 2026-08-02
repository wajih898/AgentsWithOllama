"""
Example 3: VISION AGENT — the model looks at images.

Point it at any image file (screenshot, photo, receipt, diagram,
handwritten note) and ask questions about it. Useful for:
  - Describing photos
  - Reading text from receipts/screenshots (lightweight OCR)
  - Explaining charts and diagrams

Setup:
  ollama pull llava            (or: ollama pull llama3.2-vision, needs ~8GB RAM)
  python -m pip install ollama
  python vision_agent.py path/to/image.jpg
"""

import sys
import threading
import time
from pathlib import Path
import ollama

MODEL = "llava"  # change to "llama3.2-vision" if you pulled that


def show_loading(message: str, stop_event: threading.Event) -> None:
    frames = ["|", "/", "-", "\\"]
    index = 0
    percent = 0
    while not stop_event.is_set():
        print(f"\r{message} {frames[index % len(frames)]} {percent:3d}%", end="", flush=True)
        index += 1
        percent = min(99, percent + 1)
        time.sleep(0.1)
    print(f"\r{message} ✓ 100%" + " " * 10)
    print("\r" + " " * 40 + "\r", end="", flush=True)


def ask_about_image(image_path: str, question: str) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": question,
                "images": [image_path],  # <-- just pass the file path
            }
        ],
    )
    return response["message"]["content"]


def ask_with_loading(image_path: str, question: str) -> str:
    stop_event = threading.Event()
    spinner_thread = threading.Thread(
        target=show_loading,
        args=("Analyzing image...", stop_event),
        daemon=True,
    )
    spinner_thread.start()
    try:
        return ask_about_image(image_path, question)
    finally:
        stop_event.set()
        spinner_thread.join(timeout=0.2)


if __name__ == "__main__":
    # Get image path from command line or ask for it
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Path to an image file: ").strip().strip('"').strip("'")

    if not Path(image_path).exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    # First, auto-describe the image
    print("\nAnalyzing image...")
    description = ask_with_loading(image_path, "Describe this image in detail.")
    print(f"\nDESCRIPTION:\n{description}\n")

    # Then interactive Q&A about it
    print("\nAsk questions about the image ('exit' to quit):\n")
    while True:
        q = input("Question: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit", "bye"):
            break
        print(f"\nAnswer: {ask_with_loading(image_path, q)}\n")