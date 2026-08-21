"""Structural pruning for chat transcripts (OpenAI- and Anthropic-shaped).

A document can be cut at any paragraph boundary, but a transcript can't: drop
the assistant message that called a tool and the tool's result becomes junk,
drop the result and the next assistant reply refers to output that is no
longer there. So pruning here works on whole "turns" -- a user message and
everything that follows it up to the next user message -- rather than on
individual messages.

One wrinkle: in the Anthropic shape, a tool result comes back inside a
*user*-role message (a list of ``tool_result`` content blocks). That message
must not be treated as the start of a new turn, or the result would be
grouped away from the call it answers.
"""

import json

from .tokens import estimate_tokens

__all__ = [
    "Message",
    "PruneResult",
    "parse_messages",
    "prune_messages",
    "message_text",
    "to_dicts",
]


def message_text(content):
    """Return a plain-text rendering of a message's ``content`` field.

    ``content`` may be a plain string (OpenAI shape) or a list of content
    blocks (Anthropic shape). Non-text blocks are rendered to a short
    stand-in so they still contribute to the token estimate instead of
    silently costing nothing.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                parts.append(block.get("text", ""))
            elif kind == "tool_use":
                parts.append(
                    "%s(%s)" % (block.get("name", "tool"), json.dumps(block.get("input", {}), sort_keys=True))
                )
            elif kind == "tool_result":
                inner = block.get("content", "")
                parts.append(inner if isinstance(inner, str) else json.dumps(inner, sort_keys=True))
            else:
                parts.append(json.dumps(block, sort_keys=True))
        return "\n".join(parts)
    return str(content)


def _issued_tool_call_ids(raw):
    """Tool call ids an assistant message hands out (OpenAI + Anthropic shape)."""
    ids = set()
    for call in raw.get("tool_calls") or []:
        if isinstance(call, dict) and call.get("id"):
            ids.add(call["id"])
    content = raw.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id"):
                ids.add(block["id"])
    return ids


def _referenced_tool_call_ids(raw):
    """Tool call ids a message answers (an OpenAI ``tool`` message, or
    Anthropic ``tool_result`` blocks)."""
    ids = set()
    if raw.get("role") == "tool" and raw.get("tool_call_id"):
        ids.add(raw["tool_call_id"])
    content = raw.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id"):
                ids.add(block["tool_use_id"])
    return ids


class Message(object):
    """One entry of a chat transcript, kept close to its original JSON shape."""

    __slots__ = ("role", "content", "raw", "index", "tokens")

    def __init__(self, raw, index=0):
        if not isinstance(raw, dict) or "role" not in raw:
            raise ValueError("message %d is missing a 'role'" % index)
        self.raw = raw
        self.role = raw["role"]
        self.content = raw.get("content", "")
        self.index = index
        tokens = estimate_tokens(message_text(self.content))
        tool_calls = raw.get("tool_calls")
        if tool_calls:
            tokens += estimate_tokens(json.dumps(tool_calls, sort_keys=True))
        self.tokens = tokens

    def to_dict(self):
        return dict(self.raw)

    def __repr__(self):
        return "Message(index=%d, role=%s, tokens=%d)" % (self.index, self.role, self.tokens)


class PruneResult(object):
    """The output of :func:`prune_messages`: the kept transcript plus what happened."""

    __slots__ = (
        "messages",
        "original_tokens",
        "final_tokens",
        "messages_in",
        "messages_out",
        "pinned_tool_results",
        "notes",
    )

    def __init__(
        self,
        messages,
        original_tokens,
        final_tokens,
        messages_in,
        messages_out,
        pinned_tool_results,
        notes,
    ):
        self.messages = messages
        self.original_tokens = original_tokens
        self.final_tokens = final_tokens
        self.messages_in = messages_in
        self.messages_out = messages_out
        self.pinned_tool_results = pinned_tool_results
        self.notes = notes

    def __repr__(self):
        return "PruneResult(%d -> %d tokens, %d/%d messages)" % (
            self.original_tokens,
            self.final_tokens,
            self.messages_out,
            self.messages_in,
        )


def parse_messages(data):
    """Parse a JSON chat transcript into a list of :class:`Message`.

    ``data`` may already be a list of message dicts, or a JSON string/bytes
    holding one.
    """
    if isinstance(data, (str, bytes)):
        data = json.loads(data)
    if not isinstance(data, list):
        raise ValueError("messages must be a JSON array")
    return [Message(raw, index=i) for i, raw in enumerate(data)]


def to_dicts(messages):
    """Convert a list of :class:`Message` back to plain dicts, ready for ``json.dump``."""
    return [m.to_dict() for m in messages]


def _starts_turn(message):
    """True if ``message`` opens a new user turn.

    A ``user``-role message only counts if it carries real user content. An
    Anthropic-shaped message whose content is entirely ``tool_result``
    blocks is the continuation of the turn that issued the call, not a new
    one.
    """
    if message.role != "user":
        return False
    content = message.content
    if isinstance(content, list) and content:
        return not all(
            isinstance(block, dict) and block.get("type") == "tool_result" for block in content
        )
    return True


def _marker_message(count):
    text = "[%d earlier message%s elided]" % (count, "" if count == 1 else "s")
    return Message({"role": "system", "content": text}, index=-1)


def prune_messages(messages, budget, recent_turns=2, marker=True):
    """Prune a chat transcript to fit inside ``budget`` estimated tokens.

    System messages are always kept. The transcript is otherwise split into
    turns -- a user message plus everything up to the next one -- and the
    most recent ``recent_turns`` turns are always kept whole. Older turns are
    added back, most recent first, while they still fit the budget.

    A tool call and the message holding its result are never separated: if
    selection would keep one side and drop the other, the dropped side's
    turn is pinned back in and its tool call id is recorded in
    ``pinned_tool_results``. This can push the result over budget slightly,
    which is preferred over shipping a dangling tool call or an orphaned
    result.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    if recent_turns < 0:
        raise ValueError("recent_turns must be >= 0")

    messages_in = len(messages)
    if not messages:
        return PruneResult(
            messages=[],
            original_tokens=0,
            final_tokens=0,
            messages_in=0,
            messages_out=0,
            pinned_tool_results=[],
            notes=[],
        )

    original_tokens = sum(m.tokens for m in messages)

    tags = []
    turn_members = []
    started = False
    for i, m in enumerate(messages):
        if m.role == "system":
            tags.append(("system", None))
            continue
        if not started or _starts_turn(m):
            turn_members.append([])
            started = True
        turn_members[-1].append(i)
        tags.append(("turn", len(turn_members) - 1))

    total_turns = len(turn_members)
    turn_tokens = [sum(messages[i].tokens for i in members) for members in turn_members]
    system_tokens = sum(messages[i].tokens for i, tag in enumerate(tags) if tag[0] == "system")

    recent_count = min(recent_turns, total_turns)
    kept_turns = set(range(total_turns - recent_count, total_turns))

    used = system_tokens + sum(turn_tokens[t] for t in kept_turns)
    for turn_idx in range(total_turns - recent_count - 1, -1, -1):
        cost = turn_tokens[turn_idx]
        if used + cost <= budget:
            kept_turns.add(turn_idx)
            used += cost

    issued = [set() for _ in range(total_turns)]
    referenced = [set() for _ in range(total_turns)]
    for turn_idx, members in enumerate(turn_members):
        for i in members:
            issued[turn_idx] |= _issued_tool_call_ids(messages[i].raw)
            referenced[turn_idx] |= _referenced_tool_call_ids(messages[i].raw)

    call_turn = {}
    result_turn = {}
    for turn_idx in range(total_turns):
        for call_id in issued[turn_idx]:
            call_turn[call_id] = turn_idx
        for call_id in referenced[turn_idx]:
            result_turn.setdefault(call_id, turn_idx)

    pinned_tool_results = []
    changed = True
    while changed:
        changed = False
        for call_id, call_turn_idx in call_turn.items():
            result_turn_idx = result_turn.get(call_id)
            if result_turn_idx is None or result_turn_idx == call_turn_idx:
                continue
            call_kept = call_turn_idx in kept_turns
            result_kept = result_turn_idx in kept_turns
            if call_kept and not result_kept:
                kept_turns.add(result_turn_idx)
                pinned_tool_results.append(call_id)
                changed = True
            elif result_kept and not call_kept:
                kept_turns.add(call_turn_idx)
                pinned_tool_results.append(call_id)
                changed = True

    out = []
    dropped_run = 0
    for i, message in enumerate(messages):
        tag_kind, turn_idx = tags[i]
        if tag_kind == "system" or turn_idx in kept_turns:
            if dropped_run:
                if marker:
                    out.append(_marker_message(dropped_run))
                dropped_run = 0
            out.append(message)
        else:
            dropped_run += 1
    if dropped_run and marker:
        out.append(_marker_message(dropped_run))

    notes = []
    if pinned_tool_results:
        notes.append(
            "pinned %d tool result message(s) split across a dropped turn"
            % len(sorted(set(pinned_tool_results)))
        )

    return PruneResult(
        messages=out,
        original_tokens=original_tokens,
        final_tokens=sum(m.tokens for m in out),
        messages_in=messages_in,
        messages_out=len(out),
        pinned_tool_results=sorted(set(pinned_tool_results)),
        notes=notes,
    )
