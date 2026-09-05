import pytest

from ctx_squeeze.dedupe import (
    dedupe_segments,
    find_near_duplicates,
    jaccard,
    normalize_words,
    shingles,
)
from ctx_squeeze.segments import Segment

# 14 words; one more word is appended to make 15, so a 5-word shingle window
# slides across 11 positions and only the last one touches the final word.
_STEM = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi"


def test_normalize_words_lowercases_and_strips_punctuation():
    assert normalize_words("Hello, World! 123") == ["hello", "world", "123"]


def test_shingles_of_a_short_text_is_a_single_shingle():
    assert shingles("a b c", size=5) == {("a", "b", "c")}


def test_shingles_of_a_longer_text_slides_a_window():
    result = shingles("a b c d e f", size=5)
    assert result == {("a", "b", "c", "d", "e"), ("b", "c", "d", "e", "f")}


def test_shingles_rejects_a_non_positive_size():
    with pytest.raises(ValueError):
        shingles("a b c", size=0)


def test_jaccard_edge_cases():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({1}, set()) == 0.0
    assert jaccard({1, 2}, {2, 3}) == pytest.approx(1.0 / 3.0)


def test_find_near_duplicates_flags_a_single_changed_word():
    seg_a = Segment(_STEM + " omicron", 1, 1, index=0)
    seg_b = Segment(_STEM + " pi", 1, 1, index=1)
    seg_c = Segment("zzz yyy xxx www vvv uuu ttt sss rrr", 1, 1, index=2)

    duplicates = find_near_duplicates([seg_a, seg_b, seg_c])
    assert len(duplicates) == 1
    dup_position, kept_position, score = duplicates[0]
    assert (dup_position, kept_position) == (1, 0)
    assert score == pytest.approx(10.0 / 12.0)


def test_find_near_duplicates_respects_a_stricter_threshold():
    seg_a = Segment(_STEM + " omicron", 1, 1, index=0)
    seg_b = Segment(_STEM + " pi", 1, 1, index=1)
    assert find_near_duplicates([seg_a, seg_b], threshold=0.9) == []


def test_dedupe_segments_drops_the_later_duplicate():
    seg_a = Segment(_STEM + " omicron", 1, 1, index=0)
    seg_b = Segment(_STEM + " pi", 1, 1, index=1)
    seg_c = Segment("zzz yyy xxx www vvv uuu ttt sss rrr", 1, 1, index=2)

    kept, dropped = dedupe_segments([seg_a, seg_b, seg_c])
    assert dropped == [1]
    assert [s.index for s in kept] == [0, 2]
