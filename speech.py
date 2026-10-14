import difflib
import io
import os
import re
import time
import wave

import jellyfish
import numpy as np
from dotenv import load_dotenv
from groq import Groq, InternalServerError, RateLimitError

from vocab import words_of

load_dotenv()

# Swap models here.
WHISPER_MODEL = "whisper-large-v3-turbo"
CORRECTION_MODEL = "openai/gpt-oss-20b"
LANGUAGE = "en"

# Silence gate: runs before Whisper, because Whisper "hears" its own prompt in silence.
MIN_DURATION_SEC = 0.5     # shorter recordings are treated as no speech
SILENCE_DBFS = -45.0       # a frame louder than this counts as sound (speech measured -19 to -27)
FRAME_SEC = 0.05           # loudness is measured in 50 ms frames
MIN_LOUD_FRAMES = 4        # need at least this many loud frames (0.2 s) to call Whisper

# Transcript checks.
MIN_TRANSCRIPT_LETTERS = 2  # fewer letters than this counts as no speech
ECHO_VOCAB_SHARE = 0.8      # if this share of the words are vocab terms, it's Whisper echoing its prompt
ECHO_MIN_WORDS = 3          # ...but only for 3+ words, so a one-word question like "paging" isn't rejected

# Correction safety check: keep the LLM's version only if it stayed close to Whisper's.
MIN_SIMILARITY = 0.75
MIN_LENGTH_RATIO = 0.7
MAX_LENGTH_RATIO = 1.4
# Each replaced word/span must look alike OR sound alike (Metaphone) to its replacement.
# Measured: good fixes score >= 0.67 on one of them ("red"->"thread" sounds 0.80),
# bad ones <= 0.36 on both ("red"->"deadlock" 0.36 / 0.33).
MIN_CHAR_SIMILARITY = 0.6
MIN_PHONETIC_SIMILARITY = 0.6

# Retries, same approach as OCR (to be merged into one helper in Phase 5).
MAX_RETRIES = 4            # Whisper: waiting 2, 4, 8, 16 seconds
CORRECTION_MAX_RETRIES = 1 # correction is optional, so fall back to Whisper's text quickly
MAX_RETRY_WAIT = 60        # if Groq asks us to wait longer than this, give up instead

# max_retries=0 turns off the SDK's own retries, so only our loop retries.
client = Groq(api_key=os.getenv("GROQ_API_KEY"), max_retries=0)

# Instructions, terms and examples go in the system message; the user message holds only
# the transcript, wrapped in <transcript> tags, so the model can't mistake what to correct.
CORRECTION_SYSTEM_PROMPT = """You fix speech-recognition errors in a student's spoken question about a textbook.
The user message contains only the transcript, inside <transcript>...</transcript>.
Some words may be misheard versions of these technical terms from the textbook:
{terms}

Rules:
- Only replace a word or phrase if it sounds like one of the terms above and that term fits the question.
- Use the exact spelling from the list, never expand or explain a term.
- Do not rephrase, reorder, add or remove any other words.
- Do not answer the question.
- If nothing needs fixing, return the transcript exactly as it is.
Output only the corrected transcript, without the tags, quotes or any explanation.

Examples (from other textbooks):
<transcript>how does the colonel handle system calls</transcript>
how does the kernel handle system calls

<transcript>why is a cash faster than main memory</transcript>
why is a cache faster than main memory

<transcript>what is stored in an eye node</transcript>
what is stored in an inode

<transcript>how does virtual memory work</transcript>
how does virtual memory work"""


class NoSpeechError(Exception):
    """The recording had no usable speech: too short, too quiet, empty, or an echo of the prompt."""


def _retry_delay(error, attempt):
    """Uses Groq's Retry-After header if it sent one, else 2, 4, 8, 16 seconds."""
    retry_after = error.response.headers.get("retry-after") if error.response is not None else None
    try:
        return float(retry_after)
    except (TypeError, ValueError):
        return 2 ** (attempt + 1)


def _with_retries(make_request, max_retries=MAX_RETRIES):
    """Runs make_request(), retrying per-minute rate limits (429) and server overload (503)."""
    for attempt in range(max_retries + 1):
        try:
            return make_request()
        except (RateLimitError, InternalServerError) as e:
            # A daily limit only resets over hours, so retrying now is pointless.
            if isinstance(e, RateLimitError) and " per day" in str(e).lower():
                raise
            delay = _retry_delay(e, attempt)
            if attempt == max_retries or delay > MAX_RETRY_WAIT:
                raise
            reason = "Rate limited by Groq" if isinstance(e, RateLimitError) else "Groq is over capacity"
            print(f"{reason}, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)


def has_speech(audio_bytes):
    """
    Silence gate: True if the WAV is long enough and has at least MIN_LOUD_FRAMES
    frames louder than SILENCE_DBFS. Prints the numbers, for calibrating with a real mic.
    If the file can't be read as WAV, the gate is skipped (Whisper and the echo guard still run).
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as w:
            rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as e:
        print(f"Voice input: couldn't read WAV ({e}), skipping the silence gate")
        return True

    if width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128) / 128
    elif width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768
    elif width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648
    else:
        print(f"Voice input: unsupported {width * 8}-bit WAV, skipping the silence gate")
        return True
    if channels > 1:
        samples = samples[: len(samples) - len(samples) % channels].reshape(-1, channels).mean(axis=1)

    duration = len(samples) / rate
    frame_len = max(1, int(rate * FRAME_SEC))
    n_frames = len(samples) // frame_len
    frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len) if n_frames else np.zeros((0, 1))
    rms = np.sqrt(np.mean(frames ** 2, axis=1)) if n_frames else np.array([])
    with np.errstate(divide="ignore"):
        frame_dbfs = 20 * np.log10(rms)
    loud_frames = int(np.sum(frame_dbfs > SILENCE_DBFS))
    loudest = float(frame_dbfs.max()) if n_frames else float("-inf")

    print(f"Voice input: {duration:.2f}s, loudest frame {loudest:.1f} dBFS, "
          f"{loud_frames} frames above {SILENCE_DBFS:.0f} dBFS (need {MIN_LOUD_FRAMES})")
    return duration >= MIN_DURATION_SEC and loud_frames >= MIN_LOUD_FRAMES


def transcribe_audio(audio_bytes, vocab):
    """Speech to text with Whisper, primed with the document's terms. API errors propagate."""
    response = _with_retries(lambda: client.audio.transcriptions.create(
        file=("question.wav", audio_bytes),
        model=WHISPER_MODEL,
        language=LANGUAGE,
        temperature=0,
        prompt="Glossary: " + ", ".join(vocab),
    ))
    return response.text.strip()


def looks_like_echo(text, vocab):
    """
    In silence or noise, Whisper often outputs its own prompt ("Glossary, paging, mutex").
    Treat a transcript as an echo if it says "Glossary" or is mostly vocab words.
    """
    if "glossary" in text.lower():
        return True
    words = words_of(text)
    if not words:
        return True
    if len(words) < ECHO_MIN_WORDS:
        return False
    vocab_words = {word for term in vocab for word in words_of(term)}
    return sum(word in vocab_words for word in words) / len(words) >= ECHO_VOCAB_SHARE


def _is_safe_correction(original, corrected, doc_words):
    """
    Rejects corrections that changed too much (rephrasing, answering, adding words), and
    any that introduce a word the textbook doesn't use or that is an English stop word.
    Example: "how does paging work" -> "how does it work" is rejected, because the new
    word "it" is a stop word; "what is a red" -> "what is a thread" is allowed.
    """
    if not corrected:
        return False
    similarity = difflib.SequenceMatcher(None, original.lower(), corrected.lower()).ratio()
    length_ratio = len(corrected) / len(original)
    if not (similarity >= MIN_SIMILARITY and MIN_LENGTH_RATIO <= length_ratio <= MAX_LENGTH_RATIO):
        return False

    new_words = set(words_of(corrected)) - set(words_of(original))
    unknown = new_words - doc_words
    if unknown:
        print(f"Voice input: correction introduced words not allowed from this textbook: {sorted(unknown)}")
        return False
    return _replacements_sound_alike(original, corrected)


def _replacements_sound_alike(original, corrected):
    """
    Lines up the two texts word by word, and checks every replaced span against its
    replacement. Spans are joined without spaces, so "mute ex" -> "mutex" compares
    "muteex" with "mutex". A replacement passes if it looks alike (characters) OR
    sounds alike (Metaphone codes). Adding or removing words without a counterpart fails.
    """
    before, after = words_of(original), words_of(corrected)
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        old, new = "".join(before[i1:i2]), "".join(after[j1:j2])
        if not old or not new:
            print(f"Voice input: correction added or removed words: {old or new!r}")
            return False
        char_sim = difflib.SequenceMatcher(None, old, new).ratio()
        sound_sim = difflib.SequenceMatcher(None, jellyfish.metaphone(old), jellyfish.metaphone(new)).ratio()
        if char_sim < MIN_CHAR_SIMILARITY and sound_sim < MIN_PHONETIC_SIMILARITY:
            print(f"Voice input: {old!r} -> {new!r} doesn't sound alike "
                  f"(characters {char_sim:.2f}, sound {sound_sim:.2f})")
            return False
    return True


def correct_transcript(text, vocab, doc_words):
    """
    Fixes misheard technical terms with a small LLM. Never raises: if the call fails
    or the result fails the safety check, Whisper's text is returned unchanged.
    doc_words: this document's words minus stop words; a correction may only introduce these.
    """
    if not vocab:
        return text
    try:
        response = _with_retries(lambda: client.chat.completions.create(
            model=CORRECTION_MODEL,
            messages=[
                {"role": "system", "content": CORRECTION_SYSTEM_PROMPT.format(terms=", ".join(vocab))},
                {"role": "user", "content": f"<transcript>{text}</transcript>"},
            ],
            temperature=0,
            reasoning_effort="low",
        ), max_retries=CORRECTION_MAX_RETRIES)
        corrected = response.choices[0].message.content or ""
        corrected = re.sub(r"</?transcript>", "", corrected).strip().strip('"').strip()
    except Exception as e:
        print(f"Voice input: correction failed ({type(e).__name__}), keeping Whisper's text")
        return text

    if corrected == text:
        return text
    if not _is_safe_correction(text, corrected, doc_words):
        print(f"Voice input: correction rejected by safety check: {corrected!r}")
        return text
    return corrected


def voice_to_question(audio_bytes, whisper_terms, correction_terms, doc_words):
    """
    Full pipeline: silence gate -> Whisper -> no-speech and echo checks -> correction.
    Returns (question, whisper_text). Raises NoSpeechError if there's nothing usable;
    Groq errors from Whisper (after retries) propagate to the caller.
    """
    if not has_speech(audio_bytes):
        raise NoSpeechError("too short or too quiet")

    heard = transcribe_audio(audio_bytes, whisper_terms)
    print(f"Voice input: Whisper heard {heard!r}")
    if len(re.sub(r"[^A-Za-z]", "", heard)) < MIN_TRANSCRIPT_LETTERS:
        raise NoSpeechError("empty transcript")
    if looks_like_echo(heard, whisper_terms):
        raise NoSpeechError("transcript echoes the prompt")

    return correct_transcript(heard, correction_terms, doc_words), heard
