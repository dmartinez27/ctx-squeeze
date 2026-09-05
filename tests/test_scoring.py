import math

import pytest

from ctx_squeeze.scoring import (
    inverse_document_frequency,
    score_segments,
    select_by_score,
    term_frequencies,
)
from ctx_squeeze.segments import Segment, split_segments


def test_term_frequencies_drops_stopwords_and_short_words():
    counts = term_frequencies("The cat sat on the mat")
    assert counts == {"cat": 1, "sat": 1, "mat": 1}


def test_inverse_document_frequency_rewards_rarer_terms():
    documents = [{"cat": 1, "sat": 1, "mat": 1}, {"cat": 1, "dog": 1}]
    idf = inverse_document_frequency(documents)
    assert idf["cat"] == pytest.approx(math.log(1.0 + 2 / 2.0))
    assert idf["dog"] == pytest.approx(math.log(1.0 + 2 / 1.0))
    assert idf["dog"] > idf["cat"]


def test_score_segments_rewards_the_first_and_last_position():
    segments = [Segment("foo bar baz qux", i + 1, i + 1, index=i) for i in range(3)]
    scores = score_segments(segments)
    assert scores[0] == pytest.approx(scores[2])
    assert scores[0] > scores[1]


def test_score_segments_of_empty_input():
    assert score_segments([]) == []


def _sample_segments():
    text = "alpha bravo\n\ncharlie delta\n\necho foxtrot\n\ngolf hotel"
    return split_segments(text)


def test_select_by_score_keeps_everything_that_fits():
    segments = _sample_segments()
    assert select_by_score(segments, budget=1000) == segments


def test_select_by_score_always_keeps_at_least_one_segment():
    segments = _sample_segments()
    chosen = select_by_score(segments, budget=1)
    assert len(chosen) == 1
    assert chosen[0] in segments


def test_select_by_score_preserves_original_order():
    segments = _sample_segments()
    chosen = select_by_score(segments, budget=6)
    indices = [s.index for s in chosen]
    assert indices == sorted(indices)
