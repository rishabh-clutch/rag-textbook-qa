from pdf_parser import extract_pages

def chunk_text(text, chunk_size=400, overlap=50):
    """
    Splits a block of text into overlapping chunks of ~chunk_size words.
    Returns a list of chunk strings.
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start = end - overlap  # move forward, but overlap the last `overlap` words

    return chunks


def chunk_pages(pages, chunk_size=400, overlap=50):
    """
    Takes the output of extract_pages() and returns a list of chunks,
    each still tagged with its page number.
    [{"page_num": 1, "chunk": "..."}, {"page_num": 1, "chunk": "..."}, ...]
    """
    all_chunks = []

    for page in pages:
        page_chunks = chunk_text(page["text"], chunk_size, overlap)
        for chunk in page_chunks:
            all_chunks.append({
                "page_num": page["page_num"],
                "chunk": chunk
            })

    return all_chunks


if __name__ == "__main__":
    pages = extract_pages("data/sample_textbook.pdf")
    chunks = chunk_pages(pages)
    print(f"Total chunks created: {len(chunks)}")
    print(chunks[25])