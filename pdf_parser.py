import os

import fitz  # PyMuPDF

from ocr import needs_ocr, transcribe_image, PageTooDenseError

OCR_DPI = 200              # resolution for rendering a page to an image before OCR
OCR_CACHE_DIR = "ocr_cache"  # ocr_cache/<file hash>/page_<n>.txt


def extract_pages(pdf_path):
    """
    Returns a list of dicts, one per page (blank pages included):
    [{"page_num": 1, "text": "...", "needs_ocr": False}, ...]
    needs_ocr is True when get_text() gave nothing usable (scanned or garbled page).
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        pages.append({
            "page_num": page_num,
            "text": text,
            "needs_ocr": needs_ocr(text, has_image=bool(page.get_images()))
        })

    doc.close()
    return pages


def ocr_pages(pdf_path, pages, on_progress=None, cache_dir=None):
    """
    Replaces the text of every page flagged needs_ocr with a vision-model transcription.
    on_progress(done, total) is called after each page, so a UI can show progress
    without this module knowing anything about Streamlit.
    If cache_dir is given, each page's result is saved there as soon as it's read,
    so a run stopped by a rate limit continues where it left off next time.
    """
    to_ocr = [p for p in pages if p["needs_ocr"]]
    if not to_ocr:
        return pages

    doc = fitz.open(pdf_path)
    try:
        for done, page in enumerate(to_ocr, start=1):
            cache_path = os.path.join(cache_dir, f"page_{page['page_num']}.txt") if cache_dir else None

            if cache_path and os.path.exists(cache_path):
                with open(cache_path, encoding="utf-8") as f:
                    text = f.read()
            else:
                image = doc[page["page_num"] - 1].get_pixmap(dpi=OCR_DPI).tobytes("png")
                try:
                    text = transcribe_image(image)
                except PageTooDenseError as e:
                    raise PageTooDenseError(
                        f"Page {page['page_num']} is too dense to read, even when split into quarters."
                    ) from e
                if cache_path:
                    os.makedirs(cache_dir, exist_ok=True)
                    # Write then rename, so a crash never leaves a half-written page in the cache.
                    with open(cache_path + ".tmp", "w", encoding="utf-8") as f:
                        f.write(text)
                    os.replace(cache_path + ".tmp", cache_path)

            page["text"] = text
            page["needs_ocr"] = False
            if on_progress:
                on_progress(done, len(to_ocr))
    finally:
        doc.close()

    return pages


if __name__ == "__main__":
    pages = extract_pages("data/sample_textbook.pdf")
    print(f"Extracted {len(pages)} pages")
    print(pages[0])
