"""Score one arm on the eval (or probe) set and write per-example records plus a summary."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from hvt import data, harness
from hvt.taskgen import ARG_TYPES

VISIBLE = {"no_json", "bad_shape", "schema"}  # failures the harness can detect without the answer


def summarize(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        raise ValueError("no records to summarize")
    by_tool, by_kind = defaultdict(list), defaultdict(list)
    for r in records:
        by_tool[r["tool"]].append(r["correct"])
        for arg, kind in r["kinds"].items():
            by_kind[f"{ARG_TYPES[r['tool']][arg]}:{kind}"].append(arg not in r["wrong_args"])
    rate = lambda xs: round(sum(xs) / len(xs), 4)
    return {
        "n": n,
        "accuracy": rate([r["correct"] for r in records]),
        "first_attempt_accuracy": rate([r["first_correct"] for r in records]),
        "failures": dict(Counter(r["failure"] for r in records if not r["correct"])),
        "first_attempt_failures": dict(Counter(r["first_failure"] for r in records if not r["first_correct"])),
        # Retries that fixed the call, and retries that turned a visible error into a silent wrong value.
        "repaired": sum(r["correct"] and not r["first_correct"] for r in records),
        "masked": sum(r["first_failure"] in VISIBLE and r["failure"] == "wrong_value" for r in records),
        "mean_attempts": round(sum(r["attempts"] for r in records) / n, 3),
        "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in records) / n, 1),
        "mean_completion_tokens": round(sum(r["completion_tokens"] for r in records) / n, 1),
        "accuracy_by_tool": {t: rate(v) for t, v in sorted(by_tool.items())},
        "arg_accuracy_by_kind": {k: {"n": len(v), "acc": rate(v)} for k, v in sorted(by_kind.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", help="LoRA adapter dir from hvt.train")
    ap.add_argument("--mode", choices=["plain", "harness"], default="plain")
    ap.add_argument("--split", choices=["eval", "probe"], default="eval")
    ap.add_argument("--limit", type=int, help="score only the first N examples (smoke tests)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--out", required=True, help="output prefix; writes <out>.jsonl and <out>.summary.json")
    a = ap.parse_args()

    from hvt.model import HFModel
    examples = data.eval_set() if a.split == "eval" else data.probe_set()
    examples = examples[:a.limit] if a.limit else examples
    model = HFModel(a.model, a.adapter, batch_size=a.batch_size)
    start = time.perf_counter()
    records = harness.run(model, examples, a.mode, a.max_retries)
    summary = {"model": a.model, "adapter": a.adapter, "mode": a.mode, "split": a.split,
               "seconds": round(time.perf_counter() - start, 1), **summarize(records)}

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data.write_jsonl(records, f"{out}.jsonl")
    Path(f"{out}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("accuracy", "first_attempt_accuracy", "failures",
                                              "repaired", "masked", "mean_attempts", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
