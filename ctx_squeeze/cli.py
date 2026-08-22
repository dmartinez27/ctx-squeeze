"""Command-line entry point: ``ctx-squeeze`` / ``python -m ctx_squeeze.cli``."""

import argparse
import json
import sys

from .compactor import squeeze
from .messages import parse_messages, prune_messages, to_dicts

__all__ = ["main", "build_parser"]


def _read_input(path):
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _write_output(path, text):
    if not text.endswith("\n"):
        text += "\n"
    if path is None:
        sys.stdout.write(text)
    else:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ctx-squeeze",
        description="Fit a document or chat transcript into an LLM token budget.",
    )
    parser.add_argument("input", help="path to the file to compact, or '-' for stdin")
    parser.add_argument(
        "--budget", type=int, required=True, help="target size in estimated tokens"
    )
    parser.add_argument(
        "--strategy",
        default="score",
        help="comma-separated pipeline stages: head-tail, score, dedupe (default: score)",
    )
    parser.add_argument(
        "--head-ratio",
        type=float,
        default=0.5,
        help="share of the budget spent on the head in head-tail (default: 0.5)",
    )
    parser.add_argument(
        "--jaccard",
        type=float,
        default=0.8,
        help="similarity at which two segments count as duplicates (default: 0.8)",
    )
    parser.add_argument(
        "--shingle-size",
        type=int,
        default=5,
        help="words per shingle in the dedupe stage (default: 5)",
    )
    parser.add_argument(
        "--messages", action="store_true", help="treat the input as a JSON chat transcript"
    )
    parser.add_argument(
        "--recent-turns",
        type=int,
        default=2,
        help="user turns kept whole in --messages mode (default: 2)",
    )
    parser.add_argument(
        "--no-marker", action="store_true", help="omit the elision markers"
    )
    parser.add_argument(
        "--stats", action="store_true", help="print a token summary to stderr"
    )
    parser.add_argument(
        "--json", action="store_true", help="emit a JSON report instead of plain text"
    )
    parser.add_argument(
        "-o", "--output", metavar="PATH", help="write the result to a file instead of stdout"
    )
    return parser


def _run_document(args, text):
    result = squeeze(
        text,
        budget=args.budget,
        strategy=args.strategy,
        marker=not args.no_marker,
        head_ratio=args.head_ratio,
        jaccard=args.jaccard,
        shingle_size=args.shingle_size,
    )
    if args.stats:
        sys.stderr.write(
            "kept %d of %d segments | %d -> %d tokens (budget %d)\n"
            % (
                result.segments_out,
                result.segments_in,
                result.original_tokens,
                result.final_tokens,
                args.budget,
            )
        )
        for note in result.notes:
            sys.stderr.write(note + "\n")
    if args.json:
        report = {
            "text": result.text,
            "original_tokens": result.original_tokens,
            "final_tokens": result.final_tokens,
            "segments_in": result.segments_in,
            "segments_out": result.segments_out,
            "notes": result.notes,
        }
        return json.dumps(report, indent=2)
    return result.text


def _run_messages(args, text):
    parsed = parse_messages(text)
    result = prune_messages(
        parsed,
        budget=args.budget,
        recent_turns=args.recent_turns,
        marker=not args.no_marker,
    )
    if args.stats:
        sys.stderr.write(
            "kept %d of %d messages | %d -> %d tokens (budget %d)\n"
            % (
                result.messages_out,
                result.messages_in,
                result.original_tokens,
                result.final_tokens,
                args.budget,
            )
        )
        for note in result.notes:
            sys.stderr.write(note + "\n")
    dicts = to_dicts(result.messages)
    if args.json:
        report = {
            "messages": dicts,
            "original_tokens": result.original_tokens,
            "final_tokens": result.final_tokens,
            "messages_in": result.messages_in,
            "messages_out": result.messages_out,
            "pinned_tool_results": result.pinned_tool_results,
            "notes": result.notes,
        }
        return json.dumps(report, indent=2)
    return json.dumps(dicts, indent=2)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        text = _read_input(args.input)
    except OSError as exc:
        parser.error(str(exc))

    try:
        if args.messages:
            output = _run_messages(args, text)
        else:
            output = _run_document(args, text)
    except (ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    _write_output(args.output, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
