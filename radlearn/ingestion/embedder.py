"""
radlearn/ingestion/embedder.py
───────────────────────────────
Text embedding service using Google text-embedding-004.

Connection to system:
  - Called by ingestion/pipeline.py to embed chunks after chunking.
  - Called by ingestion/image_extractor.py to embed image captions.
  - Called by chat/engine.py to embed the user's query at retrieval time.
  - Model: models/text-embedding-004 — produces 768-dimensional float vectors.
  - These vectors are stored in ChromaDB and used for cosine similarity search.

Rate limits:
  - Google Embedding API allows ~1500 requests/min on the free tier.
  - We batch 100 texts per API call and sleep 1s between batches.
  - On 429 errors, we retry with exponential backoff (3 attempts).
"""

from __future__ import annotations

import time
import logging

import google.generativeai as genai
import tiktoken

from radlearn.config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

# Configure the Google AI client once at module load.
# genai.configure is idempotent — safe to call multiple times.
# TODO: Migrate to local embeddings (e.g., HuggingFace/sentence-transformers) 
# to reduce cloud API dependencies and quota limits.
genai.configure(api_key=GOOGLE_API_KEY)

# Tokenizer for truncation. Use cl100k_base (GPT-4 tokenizer) as a close proxy
# for the Google embedding model's tokenisation.
_tokenizer = tiktoken.get_encoding("cl100k_base")


# ── Public API ────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text strings and return a list of 768-dim float vectors.

    Processing:
        1. Truncate each text to EMBEDDING_MAX_TOKENS if needed.
        2. Split list into batches of EMBEDDING_BATCH_SIZE.
        3. Call Google embedding API per batch.
        4. Sleep 1 second between batches to respect rate limits.
        5. Retry up to 3 times on API errors with exponential backoff.

    Args:
        texts: List of strings to embed.

    Returns:
        List of float vectors, one per input text.
        Order is preserved.

    Raises:
        RuntimeError: If all retry attempts fail for a batch.
    """
    if not texts:
        return []

    truncated = [_truncate(t) for t in texts]
    all_vectors: list[list[float]] = []

    for batch_start in range(0, len(truncated), EMBEDDING_BATCH_SIZE):
        batch = truncated[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
        vectors = _embed_batch_with_retry(batch)
        all_vectors.extend(vectors)

        # Rate limit: sleep between batches to stay within 100 req/min free tier.
        # 20 texts/batch × 1 batch / 15s = 80 req/min — safely under the limit.
        if batch_start + EMBEDDING_BATCH_SIZE < len(truncated):
            time.sleep(15.0)

    return all_vectors


def embed_single(text: str) -> list[float]:
    """
    Convenience wrapper to embed a single string.

    Args:
        text: The string to embed.

    Returns:
        A single 768-dim float vector.
    """
    results = embed_texts([text])
    return results[0] if results else []


# ── Private helpers ───────────────────────────────────────────────

def _truncate(text: str) -> str:
    """Truncate text to EMBEDDING_MAX_TOKENS tokens to fit model context."""
    tokens = _tokenizer.encode(text)
    if len(tokens) <= EMBEDDING_MAX_TOKENS:
        return text
    # Decode only the first EMBEDDING_MAX_TOKENS tokens
    return _tokenizer.decode(tokens[:EMBEDDING_MAX_TOKENS])


def _embed_batch_with_retry(texts: list[str], max_attempts: int = 5) -> list[list[float]]:
    """
    Call the Google embedding API for one batch.
    On 429 errors, reads the retry_delay from the error message and waits
    exactly that long before retrying — instead of guessing with fixed backoff.

    Args:
        texts:        Batch of strings (max EMBEDDING_BATCH_SIZE).
        max_attempts: Number of retry attempts.

    Returns:
        List of float vectors.

    Raises:
        RuntimeError: After max_attempts consecutive failures.
    """
    import re
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=texts,
                task_type="retrieval_document",
            )
            embeddings = result["embedding"]

            # Validate dimension
            if embeddings and len(embeddings[0]) != EMBEDDING_DIM:
                raise ValueError(
                    f"Unexpected embedding dimension: {len(embeddings[0])} (expected {EMBEDDING_DIM})"
                )

            return embeddings

        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                error_str = str(e)
                # Try to read the retry_delay Google tells us in the 429 message
                # e.g. "retry_delay { seconds: 13 }"
                match = re.search(r'retry_delay\s*\{\s*seconds:\s*(\d+)', error_str)
                if match:
                    wait = int(match.group(1)) + 2  # add 2s buffer
                else:
                    wait = 2 ** attempt  # fallback: 2s, 4s, 8s, 16s
                logger.warning(
                    "Embedding API error (attempt %d/%d): %s — retrying in %ds",
                    attempt, max_attempts, e, wait,
                )
                time.sleep(wait)
            else:
                logger.error("Embedding API failed after %d attempts: %s", max_attempts, e)

    raise RuntimeError(
        f"Embedding API failed after {max_attempts} attempts: {last_error}"
    )

