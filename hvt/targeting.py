"""Turn probe-set errors into sampling weights for the targeted training data.

An argument is charged to its phrasing kind ("date:weekday"), so the weights say which
phrasings the model gets wrong. Failures that break the whole call (no JSON, wrong tool)
are charged to the tool.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from hvt.taskgen import ARG_TYPES, KIND_OPTIONS

WHOLE_CALL = {"no_json", "bad_shape", "wrong_tool"}


def weights_from_records(records: list[dict], floor: float = 0.05) -> dict:
    """Weight = observed error rate + floor. Kinds never probed get weight 1 (assume hard)."""
    if floor < 0:
        raise ValueError("floor must be non-negative")
    seen, wrong = defaultdict(int), defaultdict(int)
    tool_seen, tool_wrong = defaultdict(int), defaultdict(int)
    for r in records:
        tool_seen[r["tool"]] += 1
        tool_wrong[r["tool"]] += not r["correct"]
        for arg, kind in r["kinds"].items():
            key = f"{ARG_TYPES[r['tool']][arg]}:{kind}"
            seen[key] += 1
            wrong[key] += r["failure"] not in WHOLE_CALL and arg in r["wrong_args"]
    kinds = {f"{t}:{k}": (wrong[f"{t}:{k}"] / seen[f"{t}:{k}"] + floor) if seen[f"{t}:{k}"] else 1.0
             for t, opts in KIND_OPTIONS.items() for k in opts}
    tools = {t: (tool_wrong[t] / tool_seen[t] + floor) if tool_seen[t] else 1.0 for t in ARG_TYPES}
    return {"kind_weights": kinds, "tool_weights": tools}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("records", help="per-example jsonl from hvt.evaluate on the probe set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--floor", type=float, default=0.05)
    a = ap.parse_args()
    with open(a.records, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    w = weights_from_records(records, a.floor)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(w, f, indent=2)
    for k, v in sorted(w["kind_weights"].items(), key=lambda kv: -kv[1]):
        print(f"{k:22s} {v:.3f}")


if __name__ == "__main__":
    main()
