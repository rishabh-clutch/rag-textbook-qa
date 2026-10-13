import streamlit as st
import os
import re
import traceback

from pdf_parser import extract_pages
from chunker import chunk_pages
from embedder import embed_chunks
from vector_store import (
    build_index, save_index, list_available_documents,
    sanitize_filename, short_file_hash, find_document_by_hash
)
from answer_generator import generate_answer
from quiz_generator import generate_quiz

st.set_page_config(page_title="RAG Textbook Q&A", page_icon="📚")


def strip_hash(doc_name):
    """Removes the trailing _<8 hex chars> hash, e.g. 'OOP_Notes_a3f9c2d1' -> 'OOP_Notes'."""
    return re.sub(r"_[0-9a-f]{8}$", "", doc_name)


st.title("📚 RAG Textbook Q&A")
st.write("Upload a textbook PDF, then ask questions and get answers with page citations.")

# --- PDF Upload Section ---
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Show the success message saved before the last st.rerun(), then forget it.
if "flash_message" in st.session_state:
    st.success(st.session_state.pop("flash_message"))

uploaded_file = st.file_uploader(
    "Upload a textbook PDF",
    type=["pdf"],
    key=f"uploader_{st.session_state.uploader_key}"
)

if uploaded_file is not None:
    if st.button("Process PDF"):
        file_hash = short_file_hash(uploaded_file.getvalue())
        existing_doc = find_document_by_hash(file_hash)

        if existing_doc:
            # Same content already indexed (even if the filename differs) — skip the work.
            # No st.rerun() here, so this message stays on screen.
            st.info(f"This PDF was already processed as '{existing_doc}'. Select it below.")
        else:
            doc_name = f"{sanitize_filename(uploaded_file.name)}_{file_hash}"

            try:
                with st.spinner("Processing PDF — this may take a minute..."):
                    os.makedirs("data", exist_ok=True)
                    temp_path = os.path.join("data", uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    pages = extract_pages(temp_path)
                    chunks = chunk_pages(pages)

                    # Only embed and save if there's real text to work with.
                    if chunks:
                        chunks = embed_chunks(chunks)
                        index, metadata = build_index(chunks)
                        save_index(index, metadata, doc_name)
            except Exception:
                traceback.print_exc()  # full error in the terminal, for debugging
                st.error(
                    "Something went wrong while processing this PDF. It may be damaged "
                    "or password-protected. Please try a different file."
                )
            else:
                # st.rerun() stays outside the try, so the except can never swallow it.
                if not chunks:
                    # No st.rerun() here, so the error stays on screen.
                    st.error(
                        "No readable text was found in this PDF. It may be a scanned document "
                        "(images of pages). Try a PDF with selectable text."
                    )
                else:
                    st.session_state.flash_message = (
                        f"Processed {len(pages)} pages into {len(chunks)} chunks. Ready for questions!"
                    )
                    st.session_state.uploader_key += 1
                    st.rerun()

st.divider()

# --- Document Selection ---
available_docs = list_available_documents()

if not available_docs:
    st.info("No documents processed yet. Upload a PDF above to get started.")
else:
    short_names = [strip_hash(doc) for doc in available_docs]

    def display_name(doc_name):
        # Hide the hash, unless another document would show the same name.
        short = strip_hash(doc_name)
        return short if short_names.count(short) == 1 else doc_name

    selected_doc = st.selectbox("Select a document:", available_docs, format_func=display_name)

    tab1, tab2 = st.tabs(["Ask a Question", "Generate Quiz"])

    with tab1:
        question = st.text_input("Enter your question:")

        if st.button("Get Answer"):
            if question:
                with st.spinner("Searching textbook and generating answer..."):
                    answer = generate_answer(question, selected_doc)
                st.subheader("Answer")
                st.write(answer)
            else:
                st.warning("Please enter a question first.")

    with tab2:
        topic = st.text_input("Enter a topic (optional — leave blank for a general quiz):")
        num_questions = st.slider("Number of questions:", min_value=3, max_value=10, value=5)

        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz..."):
                result = generate_quiz(selected_doc, num_questions=num_questions, topic=topic if topic else None)

            if "error" in result:
                st.error(result["error"])
            else:
                st.session_state.quiz_questions = result["questions"]

        if "quiz_questions" in st.session_state:
            st.subheader("Quiz")
            for i, q in enumerate(st.session_state.quiz_questions):
                st.write(f"**Q{i+1}: {q['question']}** (Page {q['page_num']})")
                user_choice = st.radio(
                    "Select an answer:",
                    q["options"],
                    key=f"quiz_q_{i}",
                    index=None
                )
                if user_choice is not None:
                    correct_option = q["options"][q["correct_answer"]]
                    if user_choice == correct_option:
                        st.success("Correct!")
                    else:
                        st.error(f"Incorrect. The correct answer is: {correct_option}")
                st.divider()