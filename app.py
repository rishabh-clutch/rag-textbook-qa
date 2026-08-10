import streamlit as st
import os

from pdf_parser import extract_pages
from chunker import chunk_pages
from embedder import embed_chunks
from vector_store import build_index, save_index, list_available_documents
from answer_generator import generate_answer

st.set_page_config(page_title="RAG Textbook Q&A", page_icon="📚")

st.title("📚 RAG Textbook Q&A")
st.write("Upload a textbook PDF, then ask questions and get answers with page citations.")

# Track an "uploader key" that we can change to force a fresh uploader widget
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_file = st.file_uploader(
    "Upload a textbook PDF",
    type=["pdf"],
    key=f"uploader_{st.session_state.uploader_key}"
)

if uploaded_file is not None:
    if st.button("Process PDF"):
        with st.spinner("Processing PDF — this may take a minute..."):
            os.makedirs("data", exist_ok=True)
            temp_path = os.path.join("data", uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            pages = extract_pages(temp_path)
            chunks = chunk_pages(pages)
            chunks = embed_chunks(chunks)
            index, metadata = build_index(chunks)
            save_index(index, metadata, uploaded_file.name)

        st.success(f"Processed {len(pages)} pages into {len(chunks)} chunks. Ready for questions!")

        # Reset the uploader by changing its key, so the next run shows a fresh, empty uploader
        st.session_state.uploader_key += 1
        st.rerun()

st.divider()

available_docs = list_available_documents()

if not available_docs:
    st.info("No documents processed yet. Upload a PDF above to get started.")
else:
    selected_doc = st.selectbox("Select a document to ask questions about:", available_docs)

    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if question:
            with st.spinner("Searching textbook and generating answer..."):
                answer = generate_answer(question, selected_doc)
            st.subheader("Answer")
            st.write(answer)
        else:
            st.warning("Please enter a question first.")