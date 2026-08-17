"""Near-duplicate detection over segments using word shingles and Jaccard.

Long agent transcripts repeat themselves: the same file gets read three times,
the same error is pasted after every retry. Exact-match dedupe misses those
because a line number or a timestamp differs. Shingling plus Jaccard similarity
catches them without any model call.
"""

import re

__all__ = [
    "normalize_words",
    "shingles",
    "jaccard",
    "find_near_duplicates",
    "dedupe_segments",
    "DEFAULT_THRESHOLD",
    "DEFAULT_SHINGLE_SIZE",
]

DEFAULT_THRESHOLD = 0.8
DEFAULT_SHINGLE_SIZE = 5

_WORD_RE = re.compile(r"[0-9a-z]+")


def normalize_words(text):
    """Lowercase ``text`` and return its alphanumeric word list."""
    return _WORD_RE.findall(text.lower())


def shingles(text, size=DEFAULT_SHINGLE_SIZE):
    """Return the set of overlapping ``size``-word shingles for ``text``.

    Text shorter than ``size`` words yields a single shingle holding every word,
    so short segments still compare sensibly against each other.
    """
    if size < 1:
        raise ValueError("shingle size must be >= 1")
    words = normalize_words(text)
    if not words:
        return set()
    if len(words) <= size:
        return set([tuple(words)])
    return set(tuple(words[i : i + size]) for i in range(len(words) - size + 1))


def jaccard(left, right):
    """Jaccard similarity of two shingle sets, in ``[0.0, 1.0]``."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / float(len(left | right))


def find_near_duplicates(
    segments, threshold=DEFAULT_THRESHOLD, shingle_size=DEFAULT_SHINGLE_SIZE
):
    """Find segments that closely repeat an earlier segment.

    Returns a list of ``(duplicate_position, kept_position, similarity)`` tuples
    where the positions index into ``segments``. The first occurrence of a
    repeated block is always the one kept.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0.0, 1.0]")
    kept = []
    duplicates = []
    for position, segment in enumerate(segments):
        fingerprint = shingles(segment.text, shingle_size)
        best_position = None
        best_score = 0.0
        for kept_position, kept_fingerprint in kept:
            score = jaccard(fingerprint, kept_fingerprint)
            if score >= threshold and score > best_score:
                best_position = kept_position
                best_score = score
        if best_position is None:
            kept.append((position, fingerprint))
        else:
            duplicates.append((position, best_position, best_score))
    return duplicates


def dedupe_segments(
    segments, threshold=DEFAULT_THRESHOLD, shingle_size=DEFAULT_SHINGLE_SIZE
):
    """Drop near-duplicate segments, keeping the first of each cluster.

    Returns ``(kept_segments, dropped_positions)``.
    """
    duplicates = find_near_duplicates(segments, threshold, shingle_size)
    dropped = set(position for position, _, _ in duplicates)
    kept = [
        segment for position, segment in enumerate(segments) if position not in dropped
    ]
    return kept, sorted(dropped)
