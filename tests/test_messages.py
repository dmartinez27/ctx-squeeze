import pytest

from ctx_squeeze.messages import (
    Message,
    message_text,
    parse_messages,
    prune_messages,
    to_dicts,
)


def _openai_transcript(turns):
    """Build a simple OpenAI-shaped transcript with ``turns`` user/assistant pairs."""
    out = [{"role": "system", "content": "you are a helpful assistant"}]
    for i in range(turns):
        out.append({"role": "user", "content": "question %d" % i})
        out.append({"role": "assistant", "content": "answer %d" % i})
    return out


def test_message_text_plain_string():
    assert message_text("hello") == "hello"


def test_message_text_none_is_empty():
    assert message_text(None) == ""


def test_message_text_renders_tool_use_and_tool_result_blocks():
    content = [
        {"type": "text", "text": "let me check"},
        {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "cats"}},
    ]
    rendered = message_text(content)
    assert "let me check" in rendered
    assert "search(" in rendered
    assert '"q": "cats"' in rendered

    result_content = [{"type": "tool_result", "tool_use_id": "call_1", "content": "3 results"}]
    assert message_text(result_content) == "3 results"


def test_parse_messages_accepts_json_string():
    parsed = parse_messages('[{"role": "user", "content": "hi"}]')
    assert len(parsed) == 1
    assert parsed[0].role == "user"


def test_parse_messages_rejects_non_list():
    with pytest.raises(ValueError):
        parse_messages('{"role": "user"}')


def test_message_requires_role():
    with pytest.raises(ValueError):
        Message({"content": "hi"})


def test_to_dicts_round_trips():
    raw = [{"role": "user", "content": "hi"}]
    assert to_dicts(parse_messages(raw)) == raw


def test_prune_messages_of_empty_transcript():
    result = prune_messages([], budget=100)
    assert result.messages == []
    assert result.messages_in == 0
    assert result.messages_out == 0


def test_prune_messages_rejects_non_positive_budget():
    messages = parse_messages(_openai_transcript(1))
    with pytest.raises(ValueError):
        prune_messages(messages, budget=0)


def test_prune_messages_rejects_negative_recent_turns():
    messages = parse_messages(_openai_transcript(1))
    with pytest.raises(ValueError):
        prune_messages(messages, budget=100, recent_turns=-1)


def test_prune_messages_keeps_system_messages_regardless_of_budget():
    messages = parse_messages(_openai_transcript(5))
    result = prune_messages(messages, budget=1, recent_turns=0)
    roles = [m.role for m in result.messages]
    assert "system" in roles


def test_prune_messages_keeps_recent_turns_whole():
    messages = parse_messages(_openai_transcript(5))
    result = prune_messages(messages, budget=1, recent_turns=2)
    kept_text = "\n".join(message_text(m.content) for m in result.messages)
    assert "question 4" in kept_text
    assert "answer 4" in kept_text
    assert "question 3" in kept_text
    assert "answer 3" in kept_text


def test_prune_messages_adds_more_turns_when_budget_allows():
    messages = parse_messages(_openai_transcript(5))
    tight = prune_messages(messages, budget=1, recent_turns=1)
    generous = prune_messages(messages, budget=10000, recent_turns=1)
    assert generous.messages_out > tight.messages_out
    assert generous.final_tokens <= 10000


def test_prune_messages_adds_elision_marker_for_dropped_turns():
    messages = parse_messages(_openai_transcript(5))
    result = prune_messages(messages, budget=1, recent_turns=1, marker=True)
    texts = [message_text(m.content) for m in result.messages]
    assert any("elided" in t for t in texts)


def test_prune_messages_can_omit_marker():
    messages = parse_messages(_openai_transcript(5))
    result = prune_messages(messages, budget=1, recent_turns=1, marker=False)
    texts = [message_text(m.content) for m in result.messages]
    assert not any("elided" in t for t in texts)


def test_prune_messages_never_exceeds_budget_with_no_system_or_recent_turns():
    # no system message here: those are kept unconditionally, which would
    # make the budget a soft rather than a hard cap.
    transcript = []
    for i in range(20):
        transcript.append({"role": "user", "content": "question %d" % i})
        transcript.append({"role": "assistant", "content": "answer %d" % i})
    messages = parse_messages(transcript)
    for budget in (5, 50, 200):
        # marker=False: the elision marker itself isn't counted against the
        # budget, so it's excluded here to test the turn-selection logic alone.
        result = prune_messages(messages, budget=budget, recent_turns=0, marker=False)
        assert result.final_tokens <= budget


def test_anthropic_tool_result_does_not_start_a_new_turn():
    transcript = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "42"}],
        },
    ]
    messages = parse_messages(transcript)
    result = prune_messages(messages, budget=10000, recent_turns=1)
    assert result.messages_out == 1


def test_prune_messages_does_not_pin_when_call_and_result_share_a_turn():
    transcript = [
        {"role": "user", "content": "old question, ignore me"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "x"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "result"}],
        },
        {"role": "assistant", "content": "here's the answer"},
        {"role": "user", "content": "a totally separate follow-up question"},
        {"role": "assistant", "content": "a totally separate follow-up answer"},
    ]
    messages = parse_messages(transcript)
    # the call and its result both live in the older turn, so a tight budget
    # can drop that whole turn without ever needing to pin anything.
    result = prune_messages(messages, budget=1, recent_turns=1)
    assert result.pinned_tool_results == []
    kept_text = "\n".join(message_text(m.content) for m in result.messages)
    assert "old question" not in kept_text
    assert "here's the answer" not in kept_text


def test_prune_messages_pins_tool_call_when_only_result_turn_is_kept():
    transcript = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "x"}}
            ],
        },
        {"role": "user", "content": "next turn starts here"},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "result"}],
        },
        {"role": "assistant", "content": "done"},
    ]
    messages = parse_messages(transcript)
    # budget=1 keeps only the most recent turn on its own merits; the tool
    # call lives in the older turn and must be pinned back in to match it.
    result = prune_messages(messages, budget=1, recent_turns=1)
    assert "call_1" in result.pinned_tool_results
    assert any("pinned 1 tool result" in note for note in result.notes)
