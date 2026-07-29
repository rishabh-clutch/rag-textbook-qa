import fitz  # PyMuPDF

def extract_pages(pdf_path):
    """
    Returns a list of dicts, one per page:
    [{"page_num": 1, "text": "..."}, {"page_num": 2, "text": "..."}, ...]
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append({
                "page_num": page_num,
                "text": text
            })

    doc.close()
    return pages


if __name__ == "__main__":
    pages = extract_pages("data/sample_textbook.pdf")
    print(f"Extracted {len(pages)} pages")
    print(pages[0])