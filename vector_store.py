import faiss
import hashlib
import json
import numpy as np
import pickle
import os
import re

from embedder import embed_chunks
from chunker import chunk_pages
from pdf_parser import extract_pages
from vocab import extract_vocab, extract_doc_words

INDEX_DIR = "indexes"


def sanitize_filename(name):
    """
    Converts a document name into a safe filename by removing
    the .pdf extension and replacing unsafe characters with underscores.
    """
    name = os.path.splitext(name)[0]  # remove .pdf extension
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)  # replace unsafe chars
    return name


def short_file_hash(file_bytes, length=8):
    """
    Returns the first `length` characters of the SHA-256 hash of the file's bytes.
    Same content always gives the same hash, whatever the filename is.
    """
    return hashlib.sha256(file_bytes).hexdigest()[:length]


def build_index(chunks):
    if not chunks:
        raise ValueError(
            "Cannot build an index from 0 chunks. The PDF may have no extractable text "
            "(e.g. a scanned document)."
        )

    embeddings =np.array([item["embedding"] for item in chunks]).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    metadata = [
        {
            "page_num": item["page_num"],
            "chunk_index": item["chunk_index"],
            "chunk": item["chunk"]
        }
        for item in chunks
    ]

    return index, metadata


def save_index(index, metadata, doc_name):
    """
    Saves the index, metadata and speech vocabulary using a filename derived from doc_name.
    Writes to temp files first, then renames them, so a failure halfway
    never leaves a .index without its _metadata.pkl and _vocab.json.
    """
    os.makedirs(INDEX_DIR, exist_ok=True)
    safe_name = sanitize_filename(doc_name)

    index_path = os.path.join(INDEX_DIR, f"{safe_name}.index")
    metadata_path = os.path.join(INDEX_DIR, f"{safe_name}_metadata.pkl")
    vocab_path = _vocab_path(safe_name)
    # Temp names don't end in ".index", so list_available_documents ignores them.
    index_tmp = index_path + ".tmp"
    metadata_tmp = metadata_path + ".tmp"
    vocab_tmp = vocab_path + ".tmp"

    vocab = _build_vocab(metadata)

    try:
        with open(metadata_tmp, "wb") as f:
            pickle.dump(metadata, f)
        with open(vocab_tmp, "w", encoding="utf-8") as f:
            json.dump(vocab, f)
        faiss.write_index(index, index_tmp)

        # Rename the .index last: once it exists, the document is visible,
        # and its metadata and vocabulary are guaranteed to be there.
        os.replace(metadata_tmp, metadata_path)
        os.replace(vocab_tmp, vocab_path)
        os.replace(index_tmp, index_path)
    except Exception:
        for path in (index_tmp, metadata_tmp, vocab_tmp):
            if os.path.exists(path):
                os.remove(path)
        raise


def _vocab_path(safe_name):
    return os.path.join(INDEX_DIR, f"{safe_name}_vocab.json")


def _build_vocab(metadata):
    whisper_terms, correction_terms = extract_vocab(metadata)
    return {
        "whisper": whisper_terms,
        "correction": correction_terms,
        "doc_words": extract_doc_words(metadata),
    }


def load_vocab(doc_name):
    """
    Returns (whisper_terms, correction_terms, doc_words) for a document.
    Older indexes have no _vocab.json (or an older one without doc_words):
    it's computed from the saved metadata, saved (temp-then-rename) for next time, and returned.
    """
    safe_name = sanitize_filename(doc_name)
    vocab_path = _vocab_path(safe_name)

    if os.path.exists(vocab_path):
        try:
            with open(vocab_path, encoding="utf-8") as f:
                data = json.load(f)
            return data["whisper"], data["correction"], set(data["doc_words"])
        except (json.JSONDecodeError, KeyError):
            pass  # damaged or outdated file: rebuild it below

    with open(os.path.join(INDEX_DIR, f"{safe_name}_metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    vocab = _build_vocab(metadata)

    with open(vocab_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    os.replace(vocab_path + ".tmp", vocab_path)
    return vocab["whisper"], vocab["correction"], set(vocab["doc_words"])


def load_index(doc_name):
    """
    Loads the index and metadata for a specific document name.
    """
    safe_name = sanitize_filename(doc_name)

    index_path = os.path.join(INDEX_DIR, f"{safe_name}.index")
    metadata_path = os.path.join(INDEX_DIR, f"{safe_name}_metadata.pkl")

    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def list_available_documents():
    """
    Scans the indexes folder and returns a list of document names
    that have already been processed.
    """
    if not os.path.exists(INDEX_DIR):
        return []

    files = os.listdir(INDEX_DIR)
    doc_names = [f[:-6] for f in files if f.endswith(".index")]  # strip ".index"
    return doc_names


def find_document_by_hash(short_hash):
    """
    Returns the name of an already-processed document whose name ends
    with this hash, or None if this PDF hasn't been processed before.
    """
    for name in list_available_documents():
        if name.endswith(f"_{short_hash}"):
            return name
    return None


def search(query_embedding, index, metadata, top_k=3):
    # Never ask FAISS for more results than the index holds.
    k = min(top_k, index.ntotal)
    if k == 0:
        return []

    query_vector = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(query_vector, k)

    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx == -1:
            continue  # FAISS uses -1 for "no neighbour"; metadata[-1] would silently return the last chunk
        result = metadata[idx].copy()
        result["distance"] = float(distance)
        results.append(result)

    return results


if __name__ == "__main__":
    pages = extract_pages("data/sample_textbook.pdf")
    chunks = chunk_pages(pages)
    chunks = embed_chunks(chunks)

    index, metadata = build_index(chunks)
    save_index(index, metadata, "sample_textbook")
    print(f"Index built and saved for 'sample_textbook' with {index.ntotal} vectors.")

    print("\nAvailable documents:", list_available_documents())