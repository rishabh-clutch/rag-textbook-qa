import faiss
import numpy as np
import pickle

from embedder import embed_chunks
from chunker import chunk_pages
from pdf_parser import extract_pages


def build_index(chunks):
    """
    Takes chunks (each with an 'embedding' field) and builds a FAISS index.
    Returns the FAISS index and a separate metadata list (page_num, chunk_index, chunk text).
    """
    embeddings = np.array([item["embedding"] for item in chunks]).astype("float32")

    dimension = embeddings.shape[1]  # should be 384 for BGE-small
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


def save_index(index, metadata, index_path="vector_store.index", metadata_path="metadata.pkl"):
    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)


def load_index(index_path="vector_store.index", metadata_path="metadata.pkl"):
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def search(query_embedding, index, metadata, top_k=3):
    """
    Given a query embedding, returns the top_k most similar chunks.
    """
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
    save_index(index, metadata)
    print(f"Index built and saved with {index.ntotal} vectors.")

    # Test a search using the first chunk's own embedding (should match itself)
    test_query = chunks[10]["embedding"]
    results = search(test_query, index, metadata, top_k=3)

    print("\nTop 3 matches for a test query:")
    for r in results:
        print(f"Page {r['page_num']}, chunk {r['chunk_index']}, distance {r['distance']:.4f}")
        print(r["chunk"][:150], "...\n")