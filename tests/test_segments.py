from ctx_squeeze.segments import split_segments, join_segments
from ctx_squeeze.tokens import estimate_tokens


def test_split_segments_of_empty_or_blank_text():
    assert split_segments("") == []
    assert split_segments("   \n\n\t\n") == []


def test_split_segments_separates_paragraphs():
    segments = split_segments("para one\n\npara two")
    assert [s.text for s in segments] == ["para one", "para two"]
    assert [s.kind for s in segments] == ["text", "text"]
    assert (segments[0].start_line, segments[0].end_line) == (1, 1)
    assert (segments[1].start_line, segments[1].end_line) == (3, 3)


def test_split_segments_keeps_fenced_code_blocks_intact():
    text = "before\n\n```\ncode line\n```\n\nafter"
    segments = split_segments(text)
    assert [s.kind for s in segments] == ["text", "code", "text"]
    assert segments[1].text == "```\ncode line\n```"
    assert (segments[1].start_line, segments[1].end_line) == (3, 5)


def test_split_segments_handles_an_unterminated_fence():
    segments = split_segments("```\ncode\nmore")
    assert len(segments) == 1
    assert segments[0].kind == "code"
    assert segments[0].text == "```\ncode\nmore"


def test_segment_index_and_token_count():
    segments = split_segments("cat\n\ndog")
    assert [s.index for s in segments] == [0, 1]
    assert segments[0].tokens == estimate_tokens("cat")


def test_join_segments_uses_the_given_separator():
    segments = split_segments("one\n\ntwo\n\nthree")
    assert join_segments(segments) == "one\n\ntwo\n\nthree"
    assert join_segments(segments, separator=" | ") == "one | two | three"


def test_join_segments_of_empty_list():
    assert join_segments([]) == ""


def test_split_segments_keeps_tilde_fenced_code_blocks_intact():
    text = "before\n\n~~~\ncode line\n~~~\n\nafter"
    segments = split_segments(text)
    assert [s.kind for s in segments] == ["text", "code", "text"]
    assert segments[1].text == "~~~\ncode line\n~~~"


def test_split_segments_detects_an_indented_fence_marker():
    text = "before\n\n  ```\n  code\n  ```\n\nafter"
    segments = split_segments(text)
    assert [s.kind for s in segments] == ["text", "code", "text"]
    assert segments[1].text == "  ```\n  code\n  ```"


def test_split_segments_only_a_matching_fence_style_closes_a_block():
    text = "```\ncode\n~~~\nmore code\n```"
    segments = split_segments(text)
    assert len(segments) == 1
    assert segments[0].kind == "code"
    assert segments[0].text == text
