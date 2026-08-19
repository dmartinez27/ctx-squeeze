"""The squeeze() pipeline: turn split segments into a document under budget.

A strategy is a comma-separated list of stage names run left to right, each
stage taking the previous stage's kept segments as input. Only the selecting
stages (``score``, ``head-tail``) are responsible for landing under budget;
``dedupe`` just removes repeats first so there is less to rank afterward.
"""

from .dedupe import DEFAULT_SHINGLE_SIZE, DEFAULT_THRESHOLD, dedupe_segments
from .scoring import select_by_score
from .segments import split_segments
from .tokens import estimate_tokens, truncate_to_tokens

__all__ = ["SqueezeResult", "squeeze", "select_head_tail", "STRATEGIES"]

ELISION_TEMPLATE = "[%d segment%s elided]"


class SqueezeResult(object):
    """The output of :func:`squeeze`: the compacted text plus what happened."""

    __slots__ = (
        "text",
        "original_tokens",
        "final_tokens",
        "segments_in",
        "segments_out",
        "notes",
    )

    def __init__(
        self, text, original_tokens, final_tokens, segments_in, segments_out, notes
    ):
        self.text = text
        self.original_tokens = original_tokens
        self.final_tokens = final_tokens
        self.segments_in = segments_in
        self.segments_out = segments_out
        self.notes = notes

    def __repr__(self):
        return "SqueezeResult(%d -> %d tokens, %d/%d segments)" % (
            self.original_tokens,
            self.final_tokens,
            self.segments_out,
            self.segments_in,
        )


def select_head_tail(segments, budget, head_ratio=0.5):
    """Keep a prefix and a suffix of segments, splitting ``budget`` between them.

    The head gets ``head_ratio`` of the budget and the tail gets the rest. Each
    side is filled greedily in document order until the next segment would not
    fit. The first segment of the head and the last segment of the tail are
    always kept even if that alone overflows their share, the same "at least
    one" guarantee :func:`~ctx_squeeze.scoring.select_by_score` makes.
    """
    if not segments:
        return []
    if not 0.0 <= head_ratio <= 1.0:
        raise ValueError("head_ratio must be in [0.0, 1.0]")

    head_budget = int(round(budget * head_ratio))
    tail_budget = budget - head_budget

    head = []
    used = 0
    i = 0
    while i < len(segments):
        cost = segments[i].tokens + (1 if head else 0)
        if head and used + cost > head_budget:
            break
        head.append(i)
        used += cost
        i += 1

    tail = []
    used = 0
    j = len(segments) - 1
    while j >= i:
        cost = segments[j].tokens + (1 if tail else 0)
        if tail and used + cost > tail_budget:
            break
        tail.append(j)
        used += cost
        j -= 1

    chosen = sorted(set(head) | set(tail))
    return [segments[k] for k in chosen]


def _stage_dedupe(segments, budget, jaccard=DEFAULT_THRESHOLD, shingle_size=DEFAULT_SHINGLE_SIZE, **_):
    kept, dropped = dedupe_segments(segments, threshold=jaccard, shingle_size=shingle_size)
    note = None
    if dropped:
        note = "dedupe dropped %d near-duplicate segment(s)" % len(dropped)
    return kept, note


def _stage_score(segments, budget, position_bonus=0.15, **_):
    return select_by_score(segments, budget, position_bonus=position_bonus), None


def _stage_head_tail(segments, budget, head_ratio=0.5, **_):
    return select_head_tail(segments, budget, head_ratio=head_ratio), None


STRATEGIES = {
    "dedupe": _stage_dedupe,
    "score": _stage_score,
    "head-tail": _stage_head_tail,
}


def _render(all_segments, kept, budget, marker):
    kept_indices = set(segment.index for segment in kept)
    pieces = []
    gap = 0
    for segment in all_segments:
        if segment.index in kept_indices:
            if gap and marker:
                pieces.append(ELISION_TEMPLATE % (gap, "" if gap == 1 else "s"))
            gap = 0
            pieces.append(segment.text)
        else:
            gap += 1
    if gap and marker:
        pieces.append(ELISION_TEMPLATE % (gap, "" if gap == 1 else "s"))

    rendered = "\n\n".join(pieces)
    tokens = estimate_tokens(rendered)
    if tokens > budget:
        rendered = truncate_to_tokens(rendered, budget)
        tokens = estimate_tokens(rendered)
    return rendered, tokens


def squeeze(text, budget, strategy="score", marker=True, **options):
    """Compact ``text`` so it fits inside ``budget`` estimated tokens.

    ``strategy`` is a comma-separated list of stage names (see ``STRATEGIES``)
    run left to right, each stage narrowing the segment list the next stage
    sees. Recognised keyword options (``jaccard``, ``shingle_size``,
    ``head_ratio``, ``position_bonus``) are forwarded to whichever stage uses
    them; a stage that doesn't use an option ignores it.

    ``final_tokens`` is guaranteed to be at most ``budget`` -- if the kept
    segments still overflow after rendering, the result is hard-truncated as a
    last resort.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    stages = [name.strip() for name in strategy.split(",") if name.strip()]
    if not stages:
        raise ValueError("strategy must name at least one stage")
    for name in stages:
        if name not in STRATEGIES:
            raise ValueError("unknown strategy stage: %r" % name)

    original_tokens = estimate_tokens(text)
    segments = split_segments(text)
    segments_in = len(segments)

    if not segments:
        return SqueezeResult(
            text="",
            original_tokens=original_tokens,
            final_tokens=0,
            segments_in=0,
            segments_out=0,
            notes=[],
        )

    notes = []
    kept = segments
    for name in stages:
        kept, note = STRATEGIES[name](kept, budget, **options)
        if note:
            notes.append(note)
        if not kept:
            kept = [segments[0]]

    rendered, final_tokens = _render(segments, kept, budget, marker)
    return SqueezeResult(
        text=rendered,
        original_tokens=original_tokens,
        final_tokens=final_tokens,
        segments_in=segments_in,
        segments_out=len(kept),
        notes=notes,
    )
