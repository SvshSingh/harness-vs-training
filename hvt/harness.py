"""Run a model over examples, either plain (one shot) or with a validate-and-retry harness.

The harness only uses what is available at inference time: the parser and the schema
validator. It never sees the gold answer, so it can fix format and schema errors but not
wrong values that happen to be valid.
"""
from __future__ import annotations

from typing import Protocol

from hvt.scoring import score
from hvt.taskgen import Example
from hvt.tools import check, system_prompt


class Model(Protocol):
    def generate(self, conversations: list[list[dict]]) -> list[tuple[str, int, int]]:
        """Return (text, prompt_tokens, completion_tokens) per conversation."""


def feedback(errors: list[str]) -> str:
    return "Your tool call was rejected:\n" + "\n".join(f"- {e}" for e in errors) + \
        "\nReply with only the corrected JSON object."


def initial_conversation(ex: Example) -> list[dict]:
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": ex.request}]


def run(model: Model, examples: list[Example], mode: str = "plain", max_retries: int = 1) -> list[dict]:
    if mode not in ("plain", "harness"):
        raise ValueError(f"unknown mode {mode!r}")
    rounds = 1 + (max_retries if mode == "harness" else 0)
    convs = [initial_conversation(ex) for ex in examples]
    attempts: list[list[str]] = [[] for _ in examples]
    tokens = [[0, 0] for _ in examples]
    pending = list(range(len(examples)))
    for r in range(rounds):
        if not pending:
            break
        outs = model.generate([convs[i] for i in pending])
        still = []
        for i, (text, p_tok, c_tok) in zip(pending, outs):
            attempts[i].append(text)
            tokens[i][0] += p_tok
            tokens[i][1] += c_tok
            errors = check(text) if mode == "harness" and r < rounds - 1 else []
            if errors:
                convs[i] += [{"role": "assistant", "content": text}, {"role": "user", "content": feedback(errors)}]
                still.append(i)
        pending = still

    records = []
    for ex, tries, (p_tok, c_tok) in zip(examples, attempts, tokens):
        final, first = score(tries[-1], ex.tool, ex.args), score(tries[0], ex.tool, ex.args)
        records.append({
            **ex.to_dict(), "mode": mode, "outputs": tries, "attempts": len(tries),
            "prompt_tokens": p_tok, "completion_tokens": c_tok,
            "correct": final.correct, "failure": final.failure, "wrong_args": final.wrong_args,
            "first_correct": first.correct, "first_failure": first.failure,
        })
    return records
