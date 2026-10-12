"""Fixed eval and probe sets, and training sets that never overlap them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hvt.harness import initial_conversation
from hvt.taskgen import Example, generate

EVAL_SEED, EVAL_N = 1000, 600     # held-out test set, scored for every arm
PROBE_SEED, PROBE_N = 2000, 400   # base-model errors here decide the targeted data


def eval_set(n: int = EVAL_N) -> list[Example]:
    return generate(n, EVAL_SEED)


def probe_set(n: int = PROBE_N) -> list[Example]:
    return generate(n, PROBE_SEED, exclude={e.request for e in eval_set()})


def reserved() -> set[str]:
    return {e.request for e in eval_set()} | {e.request for e in probe_set()}


def to_chat(ex: Example) -> dict:
    target = json.dumps({"name": ex.tool, "arguments": ex.args})
    return {"messages": initial_conversation(ex) + [{"role": "assistant", "content": target}], **ex.to_dict()}


def train_set(n: int, seed: int, weights: dict | None = None) -> list[dict]:
    w = weights or {}
    exs = generate(n, seed, tool_weights=w.get("tool_weights"), kind_weights=w.get("kind_weights"),
                   exclude=reserved())
    return [to_chat(e) for e in exs]


def write_jsonl(rows: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a training set as chat jsonl.")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--weights", help="json from hvt.targeting; omit for uniform (random arm)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    weights = None
    if a.weights:
        with open(a.weights, encoding="utf-8") as f:
            weights = json.load(f)
    rows = train_set(a.n, a.seed, weights)
    write_jsonl(rows, a.out)
    print(f"wrote {len(rows)} examples to {a.out}")


if __name__ == "__main__":
    main()
