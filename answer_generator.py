import os
from dotenv import load_dotenv
from groq import Groq

from vector_store import load_index, search
from embedder import model  # reuse the already-loaded BGE model

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_answer(question, top_k=3):
    """
    Given a student's question, retrieves relevant chunks and generates
    an answer using the LLM, grounded in the retrieved context.
    """
   
    question_embedding = model.encode(question)

   
    index, metadata = load_index()
    results = search(question_embedding, index, metadata, top_k=top_k)

   
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


    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
  
    return response.choices[0].message.content


if __name__ == "__main__":
    
    question = "What is Big O notation?"
    answer = generate_answer(question)
    print("\nQuestion:", question)
    print("\nAnswer:\n", answer)