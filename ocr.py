import base64
import io
import os
import re
import time

import numpy as np
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from PIL import Image, ImageOps

load_dotenv()

# Swap the vision model here.
VISION_MODEL = "qwen/qwen3.8-27b"

# Groq's free tier allows ~1,000 output tokens/minute for this model (~1-2 pages/minute),
# so the app only OCRs small documents. Bigger ones go through process_pdf.py in the terminal.
MAX_OCR_PAGES = 5        # refuse uploads that need more OCR'd pages than this
MIN_TEXT_CHARS = 20      # fewer non-space characters than this counts as "nearly empty"
MIN_LETTER_RATIO = 0.5   # below this share of letters, extracted text counts as garbled
MAX_IMAGE_SIDE = 2000    # pixels; keeps images well under Groq's base64 size limit

MAX_RETRIES = 4          # rate-limit retries, waiting 2, 4, 8, 16 seconds
MAX_RETRY_WAIT = 60      # if Groq asks us to wait longer than this, give up instead
# Without max_tokens, Groq guesses each request's output size itself, and once that guess
# passes the free tier's 1,000 output tokens/minute, every request is rejected as "too large".
# Setting it below 1,000 keeps requests acceptable; a page that needs more gets cut off and split.
MAX_OUTPUT_TOKENS = 900
MAX_SPLIT_DEPTH = 2      # a cut-off page is split into halves, then quarters, but no further
SPLIT_OVERLAP = 0.03     # each half extends 3% of the height past the cut (~2 lines of text)
SEAM_SEARCH_LINES = 8    # how many lines either side of the seam to check for duplicates

# max_retries=0 turns off the SDK's own retries, so only our loop below retries.
client = Groq(api_key=os.getenv("GROQ_API_KEY"), max_retries=0)

VISION_PROMPT = """Transcribe all text in this image exactly as written.
- Keep the original wording, spelling, numbers and symbols. Do not correct, summarise or translate anything.
- Format any tables as markdown tables.
- If any part is unreadable, write [unclear] in its place instead of guessing.
- Output only the transcription, with no introduction or commentary."""


class RequestTooLargeError(Exception):
    """
    The expected output for one request is bigger than the whole per-minute token limit,
    so Groq will never accept it, however long we wait. The image must be split instead.
    """


class PageTooDenseError(Exception):
    """Even a quarter of the image needs more output than one request allows, so it can't be read on this tier."""

    def __init__(self, message="This image is too dense to read, even when split into quarters."):
        super().__init__(message)


def needs_ocr(text, has_image):
    """
    Decides whether a PDF page's extracted text is unusable.
    - Nearly empty: only worth OCR if the page has an image (a scan); otherwise it's a blank page.
    - Garbled: fewer than half the non-space characters are letters (e.g. broken font encoding).
      str.isalpha() counts letters in any language, not just English.
    """
    chars = "".join(text.split())
    if len(chars) < MIN_TEXT_CHARS:
        return has_image
    letters = sum(c.isalpha() for c in chars)
    return letters / len(chars) < MIN_LETTER_RATIO


def load_image(image_bytes):
    """Opens an image, fixes phone-photo rotation, and shrinks it to MAX_IMAGE_SIDE."""
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return img


def _to_jpeg(img):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def _split_in_half(img):
    """
    Cuts the image into a top and bottom part that overlap around the cut:
    - the cut is the emptiest (brightest) row near the middle, usually a gap between lines
    - each part extends SPLIT_OVERLAP past it, so if that row wasn't truly blank,
      any line it touches is still complete in at least one part
    The duplicated lines this creates are removed by _join_overlapping.
    """
    width, height = img.size
    band_top, band_bottom = int(height * 0.4), int(height * 0.6)
    band = np.asarray(img.convert("L").crop((0, band_top, width, band_bottom)))
    cut = band_top + int(band.mean(axis=1).argmax())

    overlap = int(height * SPLIT_OVERLAP)
    top = img.crop((0, 0, width, min(height, cut + overlap)))
    bottom = img.crop((0, max(0, cut - overlap), width, height))
    return top, bottom


def _normalize(line):
    return " ".join(line.lower().split())


def _join_overlapping(top_text, bottom_text):
    """
    Joins the transcriptions of two overlapping image parts, keeping one copy of the
    lines that were read twice. Looks for the longest run of identical lines between
    the end of the top text and the start of the bottom text, then keeps the top up
    to the end of that run and the bottom from just after it. Anything dropped either
    side of the run sits in the overlap zone, so it's also in the other part; that
    includes a line sliced at a part's edge, which the other part has in full.
    If no reliable match is found, the texts are simply joined.
    """
    top_lines, bottom_lines = top_text.splitlines(), bottom_text.splitlines()
    top_tail = [i for i, line in enumerate(top_lines) if line.strip()][-SEAM_SEARCH_LINES:]
    bottom_head = [i for i, line in enumerate(bottom_lines) if line.strip()][:SEAM_SEARCH_LINES]

    best_len, best_top_end, best_bottom_end = 0, None, None
    for a in range(len(top_tail)):
        for b in range(len(bottom_head)):
            n = 0
            while (a + n < len(top_tail) and b + n < len(bottom_head)
                   and _normalize(top_lines[top_tail[a + n]]) == _normalize(bottom_lines[bottom_head[b + n]])):
                n += 1
            if n > best_len:
                best_len, best_top_end, best_bottom_end = n, top_tail[a + n - 1], bottom_head[b + n - 1]

    # One matching line only counts if it's long enough not to be a coincidence like "(a)".
    reliable = best_len >= 2 or (best_len == 1 and len(_normalize(top_lines[best_top_end])) >= 20)
    if not reliable:
        return top_text.rstrip() + "\n" + bottom_text.lstrip()

    kept = top_lines[:best_top_end + 1] + bottom_lines[best_bottom_end + 1:]
    return "\n".join(kept).strip()


def _retry_delay(error, attempt):
    """Uses Groq's Retry-After header if it sent one, else 2, 4, 8, 16 seconds."""
    retry_after = error.response.headers.get("retry-after") if error.response is not None else None
    try:
        return float(retry_after)
    except (TypeError, ValueError):
        return 2 ** (attempt + 1)


def _call_vision_model(jpeg):
    """One vision request, retried with increasing delays on ordinary rate limits."""
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": VISION_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }]

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=messages,
                temperature=0,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            break
        except RateLimitError as e:
            # Waiting can't fix a request that is bigger than the whole per-minute limit.
            if "request too large" in str(e).lower():
                raise RequestTooLargeError(str(e)) from e
            delay = _retry_delay(e, attempt)
            if attempt == MAX_RETRIES or delay > MAX_RETRY_WAIT:
                raise
            print(f"Rate limited by Groq, retrying in {delay:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(delay)

    text = response.choices[0].message.content or ""
    # Some Qwen models think out loud first; keep only the transcription.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # "length" means the answer hit MAX_OUTPUT_TOKENS, so the transcription is incomplete.
    was_cut_off = response.choices[0].finish_reason == "length"
    return text.strip(), was_cut_off


def _transcribe(img, depth=0):
    try:
        text, was_cut_off = _call_vision_model(_to_jpeg(img))
        if not was_cut_off:
            return text
        reason = "text was cut off"
    except RequestTooLargeError:
        reason = "request too large"

    if depth >= MAX_SPLIT_DEPTH:
        raise PageTooDenseError()
    print(f"Page too dense for one request ({reason}), splitting it (level {depth + 1})")
    top, bottom = _split_in_half(img)
    return _join_overlapping(_transcribe(top, depth + 1), _transcribe(bottom, depth + 1))


def transcribe_image(image_bytes):
    """Sends one image (a photo or a rendered PDF page) to the vision model and returns its text."""
    return _transcribe(load_image(image_bytes))
