# CLAUDE.md — RAG Textbook Q&A

## About me and how I want to work
- I'm Rishabh, a 3rd-year ECE student at MNIT Jaipur, preparing for placement interviews (software / AI roles).
- I'm polishing this project to (1) make it industry-grade and (2) learn backend properly, so I can explain every part in interviews.
- **Teaching style:** point out bugs and explain what's wrong, but let me write the fix myself first. Guide with questions and hints. Only give me direct code when I ask for it or say I'm short on time.
- Explain *why* behind every change, in simple language, with small examples.
- Before editing any file, tell me what you plan to change and wait for my OK.
- Verify my reasoning rigorously and correct me when I'm wrong.
- Commit after each completed step with a clear message.

## What the project does
Upload a textbook PDF, ask questions, get answers grounded in the book with page citations. Also generates MCQ quizzes (optionally on a topic). Deployed on Streamlit Cloud.

## Current architecture (files)
- `pdf_parser.py` — PyMuPDF (`fitz`), extracts text page by page, keeps page numbers. Flags pages that need OCR; `ocr_pages` OCRs only those, caching each page in `ocr_cache/<file hash>/`.
- `ocr.py` — vision OCR via Groq `qwen/qwen3.8-27b`: garbled (< 50% letters) or nearly-empty-with-image pages; images too. `max_tokens=900`, cut-off pages split into overlapping halves/quarters; 429/503 retries; daily-limit error.
- `process_pdf.py` — terminal script for big scanned PDFs (no page limit, resumes from the cache). The app's OCR limit is 5 pages.
- `chunker.py` — fixed-size chunks per page: 400 words, 50-word overlap. Chunks never span pages.
- `embedder.py` — BGE-small-en-v1.5 via sentence-transformers (384-dim vectors), loaded at import.
- `vector_store.py` — FAISS `IndexFlatL2`; saves `indexes/<sanitized_name>.index` + `<name>_metadata.pkl` (chunk text + page numbers, same order as vectors). `sanitize_filename`, `list_available_documents`, `search`.
- `answer_generator.py` — embeds question, top-3 search, refuses if best L2 distance > 1.0 (no LLM call), else calls Groq `openai/gpt-oss-120b` with a grounding prompt + page citations.
- `quiz_generator.py` — topic given: top-15 chunks via `search()`; else random 8 chunks. LLM returns JSON list of {question, options[4], correct_answer (index), page_num}.
- `app.py` — Streamlit UI: upload + "Process PDF", document selector, tabs for Q&A and Quiz.
- `vocab.py` — per-document speech vocabulary: TF-IDF (1–2 word terms) filtered by `wordfreq` rarity → top 40 terms for Whisper, top 150 for correction, plus the document's non-stop-words. Saved as `<name>_vocab.json` next to the index (built on first use for older indexes).
- `speech.py` — voice questions: silence gate → Groq `whisper-large-v3-turbo` (vocab as prompt) → no-speech/echo checks → `openai/gpt-oss-20b` correction with safety checks; falls back to Whisper's text on any doubt.
- Secrets: `GROQ_API_KEY` in `.env` (gitignored) locally, Streamlit Secrets in the cloud.

Model history: started on `llama-3.3-70b-versatile`, switched to `openai/gpt-oss-120b` after Groq deprecated it.

## Known limitation (my debugging story)
One PDF (MicrowaveTheory) returned "not found" for content clearly in the book. Root cause: broken font encoding (ToUnicode CMap) — the page renders fine but text extracts as wrong characters (`\x01\x02\x03…`, 0% letters). Fix = OCR fallback (done in Phase 2). Groq free-tier limits found along the way: 1,000 output tokens/min, 8,000 tokens/min, 200,000 tokens/day for the vision model.

## Voice input — done
"Ask a Question" tab has `st.audio_input`. A new recording (tracked by hash, never transcribed twice) is transcribed, corrected, and put into the question box for the user to edit; it is never sent straight to the LLM.
- Silence gate before Whisper: < 0.5 s, or fewer than 4 frames (50 ms each) above −45 dBFS → "Couldn't hear a question". Needed because Whisper outputs its own prompt ("Glossary, paging, mutex…") on silence or noise. Groq's `no_speech_prob` was always 0, so it isn't used.
- Echo guard: transcript contains "Glossary", or has 3+ words and ≥ 80% are vocab words → treated as no speech.
- Correction safety checks (any failure → keep Whisper's text): similarity ≥ 0.75, length 0.7×–1.4×; new words must appear in the document and not be stop words; every replaced span must look alike or sound alike (Metaphone, threshold 0.6). Prompt: instructions + terms + few-shot examples in the system message, transcript alone in `<transcript>` tags.
- Correction scored 4/6 on the hand-made misheard set ("red"→"thread" with context, "mute ex"→"mutex", "sema four"→"semaphore"); misses fall back safely to Whisper's text.

Known limitations:
- **Homophones** pass every check: "point ing" → "pointing" vs the intended "Poynting" sound identical and "pointing" is a real word in the book.
- **One-word / no-context corrections are ambiguous**: "what is a red" could be "thread" or "read"; the model may leave it or pick the wrong one.
- **Thresholds were calibrated on synthetic audio** (Windows text-to-speech, generated silence/noise), not a real mic. The terminal prints `Voice input: …dBFS…, N frames above…` for each recording, to recalibrate `SILENCE_DBFS` / `MIN_LOUD_FRAMES` with a real mic.
- Retry loop is copied from `ocr.py` into `speech.py`; merge into one helper in Phase 5.

## Polish plan (do in order, commit at the end of each phase)
1. ✅ **Robustness fixes** (bugs below) — done
2. ✅ **OCR fallback** — detect empty/garbled pages, OCR only those — done. MicrowaveTheory is at **21/44 pages** (stopped by Groq's daily limit); finish with `python process_pdf.py "C:\Users\risha\OneDrive\Desktop\MicrowaveTheory.pdf"`, which resumes from page 22.
3. 👉 **NEXT: Evaluation set** — 20–30 questions with known answer pages; measure recall@k and MRR (baseline)
4. **Hybrid search + re-ranker** — BM25 + FAISS fused with RRF, then a cross-encoder; re-measure after each
5. **Industry-grade structure** — package layout, config file, logging, tests (pytest), type hints, one shared Groq client
6. **Backend** — FastAPI (+ PostgreSQL), then a separate frontend; Streamlit calls the API instead of functions
7. **Deploy** — Docker; backend + frontend deployed separately

## Urgent housekeeping
- [~] Remove personal data from the public repo. **Done:** `indexes/` untracked and gitignored (incl. the `RISHABH_2024UEC1447` files), `data/` gitignored. **Still open:** the stray root files `metadata.pkl` and `vector_store.index` (my resume is in them) are still tracked by git. Git history still has all of them; making the repo private is the quick fix.
- [ ] Rotate the Groq API key (it was visible in screenshots) and update `.env` + Streamlit Secrets. **Still pending.**
- [x] `requirements.txt` rewritten by hand as UTF-8 with only direct dependencies, pinned.

## Phase 1 bug list — all done
1. ✅ **Filename collisions** — fixed: indexes are named `<sanitized_name>_<first 8 chars of SHA-256 of the file>`; a re-upload of the same content is detected and skipped. The dropdown hides the hash unless two names clash.
2. ✅ **`search()` with fewer vectors than `top_k`** — fixed in `search()` for every caller: asks for at most `index.ntotal` results and skips any `-1`. `answer_generator` also guards against empty results.
3. ✅ **Scanned PDF crashes processing** — fixed: 0 chunks → friendly "no readable text" error before embedding; `build_index` raises a clear `ValueError` on an empty list. (Scans are now OCR'd, Phase 2.)
4. ✅ **Quiz output trusted blindly** — fixed: each question is validated (keys, exactly 4 options, `correct_answer` int 0–3, not bool); invalid ones dropped; one retry if fewer than half are valid.
5. ✅ **No error handling around PDF processing** — fixed: friendly errors with the traceback printed to the terminal; indexes saved atomically (temp files, `.index` renamed last); success message survives `st.rerun()` via `st.session_state`.
6. ✅ **Interview mismatch** — fixed: `chunk_pages` skips chunks under 20 words (`min_words=20`).

Smaller (Phase 5): Groq client / threshold / model name duplicated across two files; raw exception text shown to users; README still says "Day 1"; `print` debugging instead of logging.

## Environment
- Windows, PowerShell, VS Code, Python 3.11, venv at `venv\`.
- Activate venv: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then `venv\Scripts\activate`.
- Run app: `streamlit run app.py`.
