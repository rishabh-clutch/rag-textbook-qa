"""
Processes a PDF from the terminal, with no OCR page limit.
Use it for big scanned or garbled PDFs that are too slow for the app on Groq's free tier.

OCR results are saved page by page in ocr_cache/, so if Groq's rate limit stops
the run, just run the same command again and it continues where it stopped.
The finished document shows up in the app's document list.

Usage: python process_pdf.py path/to/file.pdf
"""
import os
import sys
import time

from groq import RateLimitError

from pdf_parser import extract_pages, ocr_pages, OCR_CACHE_DIR
from ocr import PageTooDenseError
from chunker import chunk_pages
from embedder import embed_chunks
from vector_store import (
    build_index, save_index, sanitize_filename, short_file_hash, find_document_by_hash
)


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python process_pdf.py path/to/file.pdf")

    pdf_path = sys.argv[1]
    if not os.path.isfile(pdf_path):
        sys.exit(f"File not found: {pdf_path}")

    with open(pdf_path, "rb") as f:
        file_hash = short_file_hash(f.read())

    existing_doc = find_document_by_hash(file_hash)
    if existing_doc:
        print(f"Already processed as '{existing_doc}'. Nothing to do.")
        return

    doc_name = f"{sanitize_filename(os.path.basename(pdf_path))}_{file_hash}"
    pages = extract_pages(pdf_path)
    ocr_count = sum(p["needs_ocr"] for p in pages)
    print(f"{len(pages)} pages, {ocr_count} need OCR.")

    if ocr_count:
        start = time.time()

        def show_progress(done, total):
            print(f"  OCR page {done}/{total}  ({time.time() - start:.0f}s elapsed)")

        try:
            ocr_pages(pdf_path, pages, on_progress=show_progress,
                      cache_dir=os.path.join(OCR_CACHE_DIR, file_hash))
        except PageTooDenseError as e:
            sys.exit(f"\n{e}\nRunning again won't help: the free OCR tier can't read that page.")
        except RateLimitError as e:
            sys.exit(
                f"\nStopped by Groq's rate limit: {e}\n"
                "Pages read so far are saved. Run the same command again to continue."
            )

    chunks = chunk_pages(pages)
    if not chunks:
        sys.exit("No readable text was found, even after OCR.")

    chunks = embed_chunks(chunks)
    index, metadata = build_index(chunks)
    save_index(index, metadata, doc_name)
    print(f"Done: saved as '{doc_name}' with {len(chunks)} chunks. It now appears in the app.")


if __name__ == "__main__":
    main()
