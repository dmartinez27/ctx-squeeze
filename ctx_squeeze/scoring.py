"""Extractive scoring: decide which segments carry the most information.

Each segment is treated as a document and the whole input as the corpus, so a
term that appears in every paragraph (boilerplate, a repeated log prefix) gets a
low IDF and stops driving selection, while terms concentrated in a few segments
pull those segments up.
"""

import math

from .dedupe import normalize_words

__all__ = [
    "STOPWORDS",
    "term_frequencies",
    "inverse_document_frequency",
    "score_segments",
    "select_by_score",
    "SEPARATOR_TOKENS",
]

SEPARATOR_TOKENS = 1

STOPWORDS = frozenset(
    """
a about after all also am an and any are as at be because been before being but
by can cannot could did do does doing done down during each either else few for
from further had has have having he her here hers him his how i if in into is it
its just me more most my no nor not of off on once only or other our out over own
same she should so some such than that the their them then there these they this
those through to too under until up very was we were what when where which while
who whom why will with would you your
""".split()
)


def term_frequencies(text):
    """Return ``{term: count}`` for the non-stopword terms in ``text``."""
    counts = {}
    for word in normalize_words(text):
        if word in STOPWORDS or len(word) < 2:
            continue
        counts[word] = counts.get(word, 0) + 1
    return counts


def inverse_document_frequency(documents):
    """Compute smoothed IDF over a list of ``{term: count}`` mappings."""
    total = len(documents)
    document_frequency = {}
    for counts in documents:
        for term in counts:
            document_frequency[term] = document_frequency.get(term, 0) + 1
    return dict(
        (term, math.log(1.0 + total / float(df)))
        for term, df in document_frequency.items()
    )


def score_segments(segments, position_bonus=0.15):
    """Score every segment by keyword density.

    The score is the sum of ``(1 + log(tf)) * idf`` over the segment's distinct
    terms, divided by the square root of its length so a long rambling block
    does not automatically outrank a dense one. The first and last segments get
    a ``position_bonus`` multiplier because openings and conclusions carry
    disproportionate context.
    """
    if not segments:
        return []
    documents = [term_frequencies(segment.text) for segment in segments]
    idf = inverse_document_frequency(documents)
    scores = []
    last = len(segments) - 1
    for position, counts in enumerate(documents):
        raw = 0.0
        for term, count in counts.items():
            raw += (1.0 + math.log(count)) * idf.get(term, 0.0)
        length = sum(counts.values())
        score = raw / math.sqrt(length + 1.0)
        if position == 0 or position == last:
            score *= 1.0 + position_bonus
        scores.append(score)
    return scores


def select_by_score(segments, budget, position_bonus=0.15, overhead=0):
    """Greedily keep the highest-scoring segments that fit inside ``budget``.

    ``overhead`` is charged once per kept segment, plus once up front. Selecting
    a segment can open at most one gap before it (and the tail can open one
    more), so charging the elision-marker cost this way is an upper bound and
    keeps the rendered output inside the budget without a second pass.

    Returns the kept segments in original document order. At least one segment
    is always returned for a non-empty input, even when it overflows the budget
    on its own -- the caller decides whether to truncate it.
    """
    if not segments:
        return []
    scores = score_segments(segments, position_bonus)
    order = sorted(range(len(segments)), key=lambda i: (-scores[i], i))
    used = overhead
    chosen = []
    for position in order:
        cost = segments[position].tokens + overhead
        if chosen:
            cost += SEPARATOR_TOKENS
        if chosen and used + cost > budget:
            continue
        chosen.append(position)
        used += cost
    if not chosen:
        chosen = [order[0]]
    return [segments[position] for position in sorted(chosen)]
