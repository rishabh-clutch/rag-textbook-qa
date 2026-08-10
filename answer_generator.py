import os
from dotenv import load_dotenv
from groq import Groq

from vector_store import load_index, search
from embedder import model

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DISTANCE_THRESHOLD = 1.0


def generate_answer(question, doc_name, top_k=3):
    if not question or not question.strip():
        return "Please enter an actual question."

    if len(question.strip().split()) < 2:
        return "Please ask a more complete question — a single word isn't enough context to search on."

    try:
        index, metadata = load_index(doc_name)
    except FileNotFoundError:
        return "This document hasn't been processed yet. Please process it first before asking questions."

    question_embedding = model.encode(question)
    results = search(question_embedding, index, metadata, top_k=top_k)

    print("Retrieved chunk distances:", [round(r["distance"], 4) for r in results])

    best_distance = results[0]["distance"]
    if best_distance > DISTANCE_THRESHOLD:
        return "I couldn't find this in the provided material. (No sufficiently relevant content was found in the textbook.)"

    context = "\n\n".join([
        f"[Page {r['page_num']}]: {r['chunk']}"
        for r in results
    ])

    prompt = f"""You are a helpful assistant answering questions based ONLY on the provided textbook context.

Context:
{context}

Question: {question}

Answer using only the information in the context above. If the answer isn't in the context, say "I couldn't find this in the provided material." Cite the page number(s) you used in your answer.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
    except Exception as e:
        return f"Something went wrong while generating the answer. Please try again in a moment. (Error: {e})"

    return response.choices[0].message.content