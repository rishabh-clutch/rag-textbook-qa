import faiss
import numpy as np
import pickle
import os
import re

from embedder import embed_chunks
from chunker import chunk_pages
from pdf_parser import extract_pages

INDEX_DIR = "indexes"


def sanitize_filename(name):
    """
    Converts a document name into a safe filename by removing
    the .pdf extension and replacing unsafe characters with underscores.
    """
    name = os.path.splitext(name)[0]  # remove .pdf extension
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)  # replace unsafe chars
    return name


def build_index(chunks):
    embeddings = np.array([item["embedding"] for item in chunks]).astype("float32")

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
    Saves the index and metadata using a filename derived from doc_name.
    """
    os.makedirs(INDEX_DIR, exist_ok=True)
    safe_name = sanitize_filename(doc_name)

    index_path = os.path.join(INDEX_DIR, f"{safe_name}.index")
    metadata_path = os.path.join(INDEX_DIR, f"{safe_name}_metadata.pkl")

    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)


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


def search(query_embedding, index, metadata, top_k=3):
    query_vector = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    results = []
    for idx, distance in zip(indices[0], distances[0]):
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