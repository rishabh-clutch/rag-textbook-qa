import json
import random
from vector_store import load_index, search
from embedder import model

from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DISTANCE_THRESHOLD = 1.0


def generate_quiz(doc_name, num_questions=5, topic=None):
    """
    Generates a multiple-choice quiz. If a topic is given, retrieves
    topic-relevant chunks using the same embedding+FAISS pipeline as Q&A.
    Otherwise, falls back to random sampling across the document.
    """
    try:
        index, metadata = load_index(doc_name)
    except FileNotFoundError:
        return {"error": "This document hasn't been processed yet."}

    if topic:
        topic_embedding = model.encode(topic)
        top_n = min(15, len(metadata))
        results = search(topic_embedding, index, metadata, top_k=top_n)

        best_distance = results[0]["distance"]
        if best_distance > DISTANCE_THRESHOLD:
            return {"error": f"'{topic}' doesn't seem to be covered in this document. Try a different topic or leave it blank for a general quiz."}

        sample_chunks = results
    else:
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
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}]
        )
    except Exception as e:
        return {"error": f"Failed to generate quiz: {e}"}

    raw_text = response.choices[0].message.content.strip()

    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start != -1 and end != -1:
        raw_text = raw_text[start:end + 1]

    try:
        quiz_data = json.loads(raw_text)
    except json.JSONDecodeError:
        return {"error": "The generated quiz wasn't in a valid format. Please try again."}

    return {"questions": quiz_data}