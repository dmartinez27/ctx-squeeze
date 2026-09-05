from ctx_squeeze.tokens import estimate_tokens, fits_budget, truncate_to_tokens


def test_empty_and_whitespace_only():
    assert estimate_tokens("") == 0
    assert estimate_tokens("   \t  ") == 0


def test_word_run_counts_four_chars_per_token_with_a_floor():
    assert estimate_tokens("cat") == 1
    assert estimate_tokens("elephant") == 2


def test_digit_run_counts_three_chars_per_token():
    assert estimate_tokens("123") == 1
    assert estimate_tokens("12345") == 2


def test_cjk_counts_one_token_per_character():
    assert estimate_tokens("中文测试") == 4


def test_newline_runs_merge_and_cost_half_a_token_each():
    assert estimate_tokens("\n\n\n") == 2


def test_symbols_cost_more_per_character_than_letters():
    assert estimate_tokens("!!!") == 2


def test_fits_budget_matches_estimate_tokens():
    assert fits_budget("cat", 1) is True
    assert fits_budget("cat", 0) is False


def test_truncate_returns_empty_for_a_non_positive_budget():
    assert truncate_to_tokens("anything", 0) == ""
    assert truncate_to_tokens("anything", -5) == ""


def test_truncate_leaves_text_alone_when_it_already_fits():
    assert truncate_to_tokens("cat", 10) == "cat"


def test_truncate_cuts_at_a_word_boundary_and_fits_the_budget():
    truncated = truncate_to_tokens("cat dog bird fish", 2)
    assert truncated == "cat dog"
    assert estimate_tokens(truncated) <= 2


def test_truncate_reserves_room_for_the_suffix():
    truncated = truncate_to_tokens("cat dog bird fish", 3, suffix="...")
    assert truncated == "cat..."
    assert estimate_tokens(truncated) <= 3
