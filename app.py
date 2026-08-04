import streamlit as st
from answer_generator import generate_answer

st.set_page_config(page_title="RAG Textbook Q&A", page_icon="📚")

st.title("📚 RAG Textbook Q&A")
st.write("Ask a question about your textbook and get an answer with page citations.")

question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if question:
        with st.spinner("Searching textbook and generating answer..."):
            answer = generate_answer(question)
        st.subheader("Answer")
        st.write(answer)
    else:
        st.warning("Please enter a question first.")