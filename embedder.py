from sentence_transformers import SentenceTransformer
from chunker import chunk_pages
from pdf_parser import extract_pages

# Load the BGE model (downloads once, then cached locally)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def embed_chunks(chunks):
    """
    Takes the output of chunk_pages() and adds an 'embedding' field to each chunk.
    """
    texts = [item["chunk"] for item in chunks]
    embeddings = model.encode(texts)

    for item, embedding in zip(chunks, embeddings):
        item["embedding"] = embedding

    return chunks


if __name__ == "__main__":
    pages = extract_pages("data/sample_textbook.pdf")
    chunks = chunk_pages(pages)
    chunks_with_embeddings = embed_chunks(chunks)

    print(f"Total chunks embedded: {len(chunks_with_embeddings)}")
    print("First chunk's page number:", chunks_with_embeddings[0]["page_num"])
    print("First chunk's embedding (first 5 numbers):", chunks_with_embeddings[0]["embedding"][:5])
    print("Embedding length:", len(chunks_with_embeddings[0]["embedding"]))