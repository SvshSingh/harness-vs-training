"""Score a model reply against the gold call and name the failure."""
from __future__ import annotations

from dataclasses import dataclass, field

from hvt.tools import extract_json, validate

FAILURES = ("no_json", "bad_shape", "wrong_tool", "schema", "wrong_value")


@dataclass
class Score:
    correct: bool
    failure: str | None = None           # one of FAILURES, None when correct
    wrong_args: list[str] = field(default_factory=list)


def _same(a, b) -> bool:
    if isinstance(b, bool) or isinstance(a, bool):
        return a is b
    if isinstance(b, (int, float)):
        return isinstance(a, (int, float)) and abs(a - b) < 1e-6
    if isinstance(b, str) and isinstance(a, str):
        return a.strip().casefold() == b.casefold()
    return a == b


def score(text: str, gold_tool: str, gold_args: dict) -> Score:
    call, _ = extract_json(text)
    if call is None:
        return Score(False, "no_json", list(gold_args))
    if not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
        return Score(False, "bad_shape", list(gold_args))
    if call["name"] != gold_tool:
        return Score(False, "wrong_tool", list(gold_args))
    args = call["arguments"]
    wrong = [a for a, v in gold_args.items() if a not in args or not _same(args[a], v)]
    if validate(call):
        return Score(False, "schema", wrong)
    return Score(not wrong, "wrong_value" if wrong else None, wrong)
