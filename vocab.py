import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from wordfreq import zipf_frequency

WHISPER_VOCAB_SIZE = 40      # Whisper's prompt is capped at ~224 tokens
CORRECTION_VOCAB_SIZE = 150  # the correction LLM has no such cap
CHUNK_OVERLAP = 50           # must match the overlap used in chunker.chunk_pages

# Zipf scale: ~7 for "the", ~5 for "memory", ~3 for "semaphore".
MAX_ZIPF = 4.0               # Whisper list: a single word must be rarer than this in everyday English
MAX_ZIPF_PHRASE = 5.0        # looser for 2-word phrases, so "smith chart" survives
MAX_ZIPF_CORRECTION = 5.5    # correction list only: lets in technical words that are also everyday
                             # words, like "thread" (4.44) or "process" (5.26), but not "time" (6.29)

EXTRA_STOP_WORDS = {
    # book structure
    "figure", "fig", "section", "chapter", "example", "examples", "exercise", "exercises", "problem",
    "problems", "page", "table", "equation", "equations", "text", "note", "result", "results",
    # generic verbs and filler that are rare enough to pass the wordfreq filter
    "allocated", "allocate", "allocates", "execute", "executes", "executed", "executing", "accessed",
    "modify", "modified", "functionality", "arbitrary", "plotted", "seen", "simplifies", "represents",
    "geometry", "integral", "typically", "respectively", "corresponding", "specified", "implemented",
    "implement", "implementing", "invoked", "invoke", "denoted", "obtained", "assume", "assumed",
    "illustrate", "illustrates", "consist", "specify", "specifying", "distinguish", "construct",
    "proceeding", "tion", "acm", "bibliographical", "terminate", "programmer", "programmers",
    # OCR formatting leftovers
    "markdown", "imgs", "jpg", "png",
}


def _clean(text):
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)  # markdown image links from OCR
    text = re.sub(r"```[a-zA-Z]*", " ", text)          # code fences the OCR sometimes adds
    text = re.sub(r"\\[a-zA-Z]+", " ", text)           # LaTeX commands from OCR: \frac, \vec, \beta ...
    return text


def _without_overlap(chunks):
    """
    Each chunk after the first on a page repeats the previous chunk's last CHUNK_OVERLAP words.
    Dropping them stops a sentence that appears once in the book from counting as "in 2 chunks".
    """
    return [
        c["chunk"] if c["chunk_index"] == 0 else " ".join(c["chunk"].split()[CHUNK_OVERLAP:])
        for c in chunks
    ]


def _singular(term):
    """Rough singular of a term's last word: 'processes' -> 'process', 'queries' -> 'query'."""
    words = term.split()
    last = words[-1]
    if last.endswith("ies") and len(last) > 4:
        last = last[:-3] + "y"
    elif last.endswith(("sses", "xes", "ches", "shes")):
        last = last[:-2]
    elif last.endswith("s") and not last.endswith("ss") and len(last) > 3:
        last = last[:-1]
    return " ".join(words[:-1] + [last])


def _rare_enough(term, word_limit=MAX_ZIPF):
    limit = MAX_ZIPF_PHRASE if " " in term else word_limit
    return min(zipf_frequency(word, "en") for word in term.split()) < limit


def words_of(text):
    """Lowercase words with punctuation removed: "What's the SWR?" -> ["what", "s", "the", "swr"]."""
    return re.findall(r"[a-z][a-z0-9]*", text.lower())


def extract_doc_words(chunks):
    """
    Every word in the document except English stop words ("it", "the", "does" ...).
    Used to check that a transcript correction only introduces words the textbook really uses.
    No rarity filter here: technical words like "thread" or "process" are also everyday words.
    """
    words = set()
    for c in chunks:
        words.update(words_of(_clean(c["chunk"])))
    return sorted(words - ENGLISH_STOP_WORDS)


def extract_vocab(chunks):
    """
    Finds a document's distinctive technical terms, for speech recognition.
    chunks: the index metadata, [{"chunk": "...", "chunk_index": 0, ...}, ...]
    Returns (whisper_terms, correction_terms): the top WHISPER_VOCAB_SIZE and
    CORRECTION_VOCAB_SIZE terms, most distinctive first.
    """
    texts = [_clean(t) for t in _without_overlap(chunks)]
    texts = [t for t in texts if t.strip()]
    if not texts:
        return [], []

    # A term must appear in 2+ chunks (filters one-off junk) and in at most half of them
    # (filters repeated page headers). Tiny documents can't support those limits.
    many_chunks = len(texts) >= 4
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=list(ENGLISH_STOP_WORDS | EXTRA_STOP_WORDS),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",  # words of 3+ letters, no pure numbers
        min_df=2 if many_chunks else 1,
        max_df=0.5 if many_chunks else 1.0,
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:  # no usable words at all
        return [], []

    scores = dict(zip(vectorizer.get_feature_names_out(), np.asarray(matrix.sum(axis=0)).ravel()))

    # Merge plurals into the singular form when both appear.
    merged = {}
    for term, score in scores.items():
        key = _singular(term) if _singular(term) in scores else term
        merged[key] = merged.get(key, 0) + score

    ranked = [t for t, _ in sorted(merged.items(), key=lambda kv: -kv[1])]
    whisper_terms = [t for t in ranked if _rare_enough(t)][:WHISPER_VOCAB_SIZE]
    correction_terms = [t for t in ranked if _rare_enough(t, MAX_ZIPF_CORRECTION)][:CORRECTION_VOCAB_SIZE]
    return whisper_terms, correction_terms
