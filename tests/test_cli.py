import json

import pytest

from ctx_squeeze.cli import build_parser, main


def _padded_paragraphs(count):
    return "\n\n".join(
        "paragraph number %d with some words to pad it out" % i for i in range(count)
    )


def test_build_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["input.txt", "--budget", "100"])
    assert args.strategy == "score"
    assert args.head_ratio == 0.5
    assert args.jaccard == 0.8
    assert args.shingle_size == 5
    assert args.recent_turns == 2
    assert args.messages is False
    assert args.no_marker is False
    assert args.stats is False
    assert args.json is False
    assert args.output is None


def test_build_parser_requires_budget():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["input.txt"])


def test_main_document_mode_writes_plain_text_to_stdout(tmp_path, capsys):
    text = _padded_paragraphs(20)
    src = tmp_path / "doc.txt"
    src.write_text(text)

    exit_code = main([str(src), "--budget", "30", "--strategy", "score"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "elided" in out
    assert out.endswith("\n")


def test_main_document_mode_json_report(tmp_path, capsys):
    text = _padded_paragraphs(20)
    src = tmp_path / "doc.txt"
    src.write_text(text)

    main([str(src), "--budget", "30", "--json"])

    report = json.loads(capsys.readouterr().out)
    assert set(report) == {
        "text",
        "original_tokens",
        "final_tokens",
        "segments_in",
        "segments_out",
        "notes",
    }
    assert report["segments_out"] < report["segments_in"]


def test_main_document_mode_stats_go_to_stderr(tmp_path, capsys):
    text = _padded_paragraphs(20)
    src = tmp_path / "doc.txt"
    src.write_text(text)

    main([str(src), "--budget", "30", "--stats"])

    err = capsys.readouterr().err
    assert "kept" in err
    assert "budget 30" in err


def test_main_document_mode_no_marker(tmp_path, capsys):
    text = _padded_paragraphs(20)
    src = tmp_path / "doc.txt"
    src.write_text(text)

    main([str(src), "--budget", "30", "--no-marker"])

    assert "elided" not in capsys.readouterr().out


def test_main_reads_from_stdin(monkeypatch, capsys):
    import io
    import sys

    monkeypatch.setattr(sys, "stdin", io.StringIO("just one short paragraph"))

    main(["-", "--budget", "1000"])

    assert capsys.readouterr().out.strip() == "just one short paragraph"


def test_main_writes_output_to_file(tmp_path):
    src = tmp_path / "doc.txt"
    src.write_text("just one short paragraph")
    dest = tmp_path / "out.txt"

    main([str(src), "--budget", "1000", "-o", str(dest)])

    assert dest.read_text() == "just one short paragraph\n"


def test_main_missing_input_file_is_a_parser_error(tmp_path, capsys):
    missing = tmp_path / "does-not-exist.txt"

    with pytest.raises(SystemExit):
        main([str(missing), "--budget", "100"])

    assert "No such file" in capsys.readouterr().err


def test_main_messages_mode_plain_output(tmp_path, capsys):
    transcript = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "hello"},
    ]
    src = tmp_path / "chat.json"
    src.write_text(json.dumps(transcript))

    main([str(src), "--budget", "1000", "--messages"])

    kept = json.loads(capsys.readouterr().out)
    assert kept == transcript


def test_main_messages_mode_json_report(tmp_path, capsys):
    transcript = [
        {"role": "user", "content": "turn one"},
        {"role": "assistant", "content": "reply one"},
        {"role": "user", "content": "turn two"},
        {"role": "assistant", "content": "reply two"},
    ]
    src = tmp_path / "chat.json"
    src.write_text(json.dumps(transcript))

    main([str(src), "--budget", "1000", "--messages", "--json", "--recent-turns", "1"])

    report = json.loads(capsys.readouterr().out)
    assert set(report) == {
        "messages",
        "original_tokens",
        "final_tokens",
        "messages_in",
        "messages_out",
        "pinned_tool_results",
        "notes",
    }
    assert report["messages_in"] == 4


def test_main_messages_mode_rejects_invalid_json(tmp_path, capsys):
    src = tmp_path / "chat.json"
    src.write_text("not json at all")

    with pytest.raises(SystemExit):
        main([str(src), "--budget", "100", "--messages"])

    assert capsys.readouterr().err
