import pytest

from ctx_squeeze.compactor import STRATEGIES, select_head_tail, squeeze
from ctx_squeeze.segments import split_segments


def _padded_paragraphs(count):
    return "\n\n".join(
        "paragraph number %d with some words to pad it out" % i for i in range(count)
    )


def test_strategies_registers_the_documented_stages():
    assert set(STRATEGIES) == {"dedupe", "score", "head-tail"}


def test_squeeze_rejects_a_non_positive_budget():
    with pytest.raises(ValueError):
        squeeze("some text", budget=0)


def test_squeeze_rejects_an_unknown_strategy():
    with pytest.raises(ValueError):
        squeeze("some text", budget=100, strategy="not-a-real-stage")


def test_squeeze_of_empty_text():
    result = squeeze("", budget=100)
    assert result.text == ""
    assert result.segments_in == 0
    assert result.segments_out == 0
    assert result.final_tokens == 0


def test_squeeze_never_exceeds_the_budget():
    text = _padded_paragraphs(20)
    for budget in (5, 20, 60, 200):
        result = squeeze(text, budget=budget, strategy="score")
        assert result.final_tokens <= budget


def test_squeeze_adds_an_elision_marker_when_segments_are_dropped():
    text = _padded_paragraphs(20)
    result = squeeze(text, budget=30, strategy="score", marker=True)
    assert "elided" in result.text
    assert result.segments_out < result.segments_in


def test_squeeze_can_omit_the_marker():
    text = _padded_paragraphs(20)
    result = squeeze(text, budget=30, strategy="score", marker=False)
    assert "elided" not in result.text


def test_squeeze_chains_dedupe_before_scoring():
    text = "\n\n".join(["identical paragraph text here"] * 3 + ["a unique closing paragraph"])
    result = squeeze(text, budget=1000, strategy="dedupe,score")
    assert result.segments_in == 4
    assert result.segments_out == 2
    assert any("dedupe" in note for note in result.notes)


def test_select_head_tail_always_keeps_the_first_and_last_segment():
    segments = split_segments(_padded_paragraphs(10))
    chosen = select_head_tail(segments, budget=1)
    indices = [s.index for s in chosen]
    assert 0 in indices
    assert segments[-1].index in indices


def test_select_head_tail_keeps_everything_when_the_budget_is_generous():
    segments = split_segments(_padded_paragraphs(10))
    assert select_head_tail(segments, budget=1000, head_ratio=0.5) == segments


def test_select_head_tail_rejects_a_bad_ratio():
    with pytest.raises(ValueError):
        select_head_tail(split_segments("a\n\nb"), budget=10, head_ratio=1.5)


def test_select_head_tail_of_empty_input():
    assert select_head_tail([], budget=10) == []
