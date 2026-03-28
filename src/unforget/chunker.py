"""Sentence-boundary chunking with overlap for conversation ingest.

Splits text at sentence boundaries with configurable overlap.
Small chunks = better semantic precision, large chunks = more context.
Overlap preserves cross-sentence meaning.
"""

from __future__ import annotations

import re

# Sentence boundary pattern: period/exclamation/question followed by space + uppercase
# or end of string. Handles common abbreviations (Mr., Dr., etc.)
_ABBREVS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc", "ltd", "corp"}
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\'])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Handles basic abbreviations."""
    raw_splits = _SENT_RE.split(text.strip())

    # Rejoin false splits from abbreviations
    sentences: list[str] = []
    buffer = ""
    for part in raw_splits:
        if buffer:
            part = buffer + " " + part
            buffer = ""

        # Check if this ends with an abbreviation
        words = part.rstrip(".!?").split()
        if words and words[-1].lower().rstrip(".") in _ABBREVS and part.rstrip()[-1] == ".":
            buffer = part
            continue

        sentences.append(part.strip())

    if buffer:
        sentences.append(buffer.strip())

    return [s for s in sentences if s]


def chunk_text(
    text: str,
    *,
    min_sentences: int = 2,
    max_sentences: int = 8,
    overlap: int = 1,
) -> list[str]:
    """Split text into overlapping chunks at sentence boundaries.

    Args:
        text: Input text to chunk.
        min_sentences: Minimum sentences per chunk.
        max_sentences: Maximum sentences per chunk.
        overlap: Number of sentences to overlap between chunks.

    Returns:
        List of text chunks.
    """
    sentences = split_sentences(text)

    if not sentences:
        return []

    # If total sentences fit in one chunk, return as-is
    if len(sentences) <= max_sentences:
        return [" ".join(sentences)]

    chunks: list[str] = []
    start = 0

    while start < len(sentences):
        end = min(start + max_sentences, len(sentences))
        chunk_sentences = sentences[start:end]

        # Enforce minimum chunk size (except for the last chunk)
        if len(chunk_sentences) < min_sentences and chunks:
            # Too small — merge with previous chunk
            break

        chunks.append(" ".join(chunk_sentences))

        # Advance with overlap
        step = max_sentences - overlap
        if step < 1:
            step = 1
        start += step

    return chunks


def chunk_messages(
    messages: list[dict[str, str]],
    *,
    min_sentences: int = 2,
    max_sentences: int = 8,
    overlap: int = 1,
) -> list[str]:
    """Chunk a conversation (list of message dicts) into text chunks.

    Concatenates all message contents with role prefixes, then chunks.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if content:
            parts.append(f"{role}: {content}")

    full_text = " ".join(parts)
    return chunk_text(
        full_text,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
        overlap=overlap,
    )
