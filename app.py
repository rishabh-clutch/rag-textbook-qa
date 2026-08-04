import streamlit as st
import os

from pdf_parser import extract_pages
from chunker import chunk_pages
from embedder import embed_chunks
from vector_store import build_index, save_index
from answer_generator import generate_answer

st.set_page_config(page_title="RAG Textbook Q&A", page_icon="📚")

st.title("📚 RAG Textbook Q&A")
st.write("Upload a textbook PDF, then ask questions and get answers with page citations.")

# --- PDF Upload Section ---
uploaded_file = st.file_uploader("Upload a textbook PDF", type=["pdf"])

if uploaded_file is not None:
    if st.button("Process PDF"):
        with st.spinner("Processing PDF — this may take a minute..."):
            # Save the uploaded file to disk temporarily
            os.makedirs("data", exist_ok=True)
            temp_path = os.path.join("data", uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Run the full pipeline
            pages = extract_pages(temp_path)
            chunks = chunk_pages(pages)
            chunks = embed_chunks(chunks)
            index, metadata = build_index(chunks)
            save_index(index, metadata)

        st.success(f"Processed {len(pages)} pages into {len(chunks)} chunks. Ready for questions!")

st.divider()

# --- Question Section ---
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if question:
        with st.spinner("Searching textbook and generating answer..."):
            answer = generate_answer(question)
        st.subheader("Answer")
        st.write(answer)
    else:
        st.warning("Please enter a question first.")