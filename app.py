import streamlit as st
import os
import re
import traceback

from groq import InternalServerError, RateLimitError

from pdf_parser import extract_pages, ocr_pages, OCR_CACHE_DIR
from ocr import transcribe_image, MAX_OCR_PAGES, PageTooDenseError, DailyLimitError
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
st.write("Upload a textbook PDF or a photo of notes, then ask questions and get answers with page citations.")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# --- PDF Upload Section ---
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Show the success message saved before the last st.rerun(), then forget it.
if "flash_message" in st.session_state:
    st.success(st.session_state.pop("flash_message"))

uploaded_file = st.file_uploader(
    "Upload a textbook PDF or an image (JPG/PNG)",
    type=["pdf", "jpg", "jpeg", "png"],
    key=f"uploader_{st.session_state.uploader_key}"
)

if uploaded_file is not None:
    if st.button("Process file"):
        file_hash = short_file_hash(uploaded_file.getvalue())
        existing_doc = find_document_by_hash(file_hash)

        if existing_doc:
            # Same content already indexed (even if the filename differs) — skip the work.
            # No st.rerun() here, so this message stays on screen.
            st.info(f"This file was already processed as '{existing_doc}'. Select it below.")
        else:
            doc_name = f"{sanitize_filename(uploaded_file.name)}_{file_hash}"
            is_image = uploaded_file.name.lower().endswith(IMAGE_EXTENSIONS)
            too_many_for_ocr = False
            chunks = []

            try:
                with st.spinner("Processing — this may take a minute..."):
                    os.makedirs("data", exist_ok=True)
                    temp_path = os.path.join("data", uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # An image is one page that always needs OCR.
                    if is_image:
                        pages = [{"page_num": 1, "text": "", "needs_ocr": True}]
                    else:
                        pages = extract_pages(temp_path)

                    ocr_count = sum(p["needs_ocr"] for p in pages)

                    # Check the limit before sending anything to the vision model.
                    if ocr_count > MAX_OCR_PAGES:
                        too_many_for_ocr = True
                    else:
                        if ocr_count:
                            slow_note = "OCR is slow on the free tier, so this may take a few minutes."
                            progress = st.progress(0.0, text=f"Reading pages with OCR: 0 of {ocr_count}. {slow_note}")

                            def show_progress(done, total):
                                progress.progress(
                                    done / total, text=f"Reading pages with OCR: {done} of {total}. {slow_note}"
                                )

                            if is_image:
                                pages[0]["text"] = transcribe_image(uploaded_file.getvalue())
                                show_progress(1, 1)
                            else:
                                ocr_pages(temp_path, pages, on_progress=show_progress,
                                          cache_dir=os.path.join(OCR_CACHE_DIR, file_hash))
                            progress.empty()

                        chunks = chunk_pages(pages)

                        # Only embed and save if there's real text to work with.
                        if chunks:
                            chunks = embed_chunks(chunks)
                            index, metadata = build_index(chunks)
                            save_index(index, metadata, doc_name)
            except PageTooDenseError as e:
                traceback.print_exc()
                # Waiting won't help here, so don't suggest trying again.
                st.error(f"{e} The free OCR tier can't read it. Try a lower-detail scan of that page.")
            except DailyLimitError as e:
                traceback.print_exc()
                st.error(
                    f"{e} Please try again tomorrow. For PDFs, pages already read are saved, "
                    "so it continues where it stopped."
                )
            except InternalServerError:
                traceback.print_exc()
                st.error("The OCR service is temporarily overloaded. Please try again in a few minutes.")
            except RateLimitError:
                traceback.print_exc()
                st.error(
                    "The OCR service's free usage limit was reached. Please wait a minute and "
                    "click Process again. Pages already read are saved, so it continues where it stopped."
                )
            except Exception:
                traceback.print_exc()  # full error in the terminal, for debugging
                st.error(
                    "Something went wrong while processing this file. It may be damaged "
                    "or password-protected. Please try a different file."
                )
            else:
                # st.rerun() stays outside the try, so the except can never swallow it.
                # The error branches don't rerun, so their messages stay on screen.
                if too_many_for_ocr:
                    st.error(
                        f"{ocr_count} pages in this PDF are scanned or unreadable and would need OCR, "
                        f"but the app's limit is {MAX_OCR_PAGES} pages (OCR is slow on the free tier). "
                        f"For bigger PDFs, run this in the terminal: python process_pdf.py \"{uploaded_file.name}\""
                    )
                elif not chunks:
                    st.error(
                        "No readable text was found in this file, even after OCR. "
                        "Try a clearer scan or photo."
                    )
                else:
                    pages_with_text = sum(1 for p in pages if p["text"].strip())
                    ocr_note = f" ({ocr_count} read with OCR)" if ocr_count else ""
                    st.session_state.flash_message = (
                        f"Processed {pages_with_text} pages{ocr_note} into {len(chunks)} chunks. "
                        "Ready for questions!"
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