import json
import random
from vector_store import load_index

from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_quiz(doc_name, num_questions=5):
    """
    Generates a multiple-choice quiz based on a random sample of chunks
    from the given document.
    """
    try:
        index, metadata = load_index(doc_name)
    except FileNotFoundError:
        return {"error": "This document hasn't been processed yet."}

    # Pick a random sample of chunks to base the quiz on
    sample_size = min(8, len(metadata))
    sample_chunks = random.sample(metadata, sample_size)

    content = "\n\n".join([
        f"[Page {c['page_num']}]: {c['chunk']}"
        for c in sample_chunks
    ])

    prompt = f"""You are creating a multiple-choice quiz based on textbook content.

Textbook content:
{content}

Generate exactly {num_questions} multiple-choice questions based ONLY on the content above.

Respond with ONLY valid JSON, in exactly this format, with no extra text before or after:
[
  {{
    "question": "question text here",
    "options": ["option A", "option B", "option C", "option D"],
    "correct_answer": 0,
    "page_num": 22
  }}
]

The "correct_answer" field must be the index (0, 1, 2, or 3) of the correct option in the "options" list.
The "page_num" field should be the page number the question was based on.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
    except Exception as e:
        return {"error": f"Failed to generate quiz: {e}"}

    raw_text = response.choices[0].message.content.strip()

    try:
        quiz_data = json.loads(raw_text)
    except json.JSONDecodeError:
        return {"error": "The generated quiz wasn't in a valid format. Please try again."}

    return {"questions": quiz_data}


if __name__ == "__main__":
    result = generate_quiz("RISHABH_2024UEC1447", num_questions=3)

    if "error" in result:
        print("Error:", result["error"])
    else:
        for i, q in enumerate(result["questions"], start=1):
            print(f"\nQ{i}: {q['question']} (Page {q['page_num']})")
            for j, option in enumerate(q["options"]):
                marker = "✓" if j == q["correct_answer"] else " "
                print(f"  [{marker}] {option}")