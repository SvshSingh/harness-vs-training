"""Collect every arm into results/table.md and results/accuracy.png."""
from __future__ import annotations

import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

RESULTS, RUNS = Path("results"), Path("runs")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PLAIN, HARNESS = "#2a78d6", "#eb6834"  # categorical slots 1 and 2


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def collect() -> list[dict]:
    """One row per arm, averaged over seeds."""
    groups: dict[str, list[dict]] = defaultdict(list)
    base = {m: _load(RESULTS / "base" / f"{m}.summary.json") for m in ("plain", "harness")}
    if all(base.values()):
        groups["base"].append({"plain": base["plain"], "harness": base["harness"], "meta": None})
    for d in sorted(RESULTS.iterdir()) if RESULTS.exists() else []:
        m = re.fullmatch(r"(random|targeted)_(\d+)_s(\d+)", d.name)
        runs = {k: _load(d / f"{k}.summary.json") for k in ("plain", "harness")}
        if m and all(runs.values()):
            groups[f"{m[1]}_{m[2]}"].append({**runs, "meta": _load(RUNS / d.name / "train_meta.json")})

    def agg(xs: list[float]) -> tuple[float, float]:
        return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)

    rows = []
    for arm, rs in groups.items():
        metas = [r["meta"] for r in rs if r["meta"]]
        rows.append({
            "arm": arm, "seeds": len(rs),
            "plain": agg([r["plain"]["accuracy"] for r in rs]),
            "harness": agg([r["harness"]["accuracy"] for r in rs]),
            "gain": agg([r["harness"]["accuracy"] - r["plain"]["accuracy"] for r in rs]),
            "harness_tokens": st.mean(r["harness"]["mean_prompt_tokens"] + r["harness"]["mean_completion_tokens"] for r in rs),
            "plain_tokens": st.mean(r["plain"]["mean_prompt_tokens"] + r["plain"]["mean_completion_tokens"] for r in rs),
            "train_tokens": st.mean(m["train_tokens"] for m in metas) if metas else 0,
            "train_seconds": st.mean(m["seconds"] for m in metas) if metas else 0,
            "plain_failures": rs[0]["plain"]["failures"],
        })
    order = lambda r: (r["arm"] != "base", int(r["arm"].split("_")[1]) if "_" in r["arm"] else 0, r["arm"])
    return sorted(rows, key=order)


def label(arm: str) -> str:
    return "Base" if arm == "base" else f"{arm.split('_')[0].capitalize()} {arm.split('_')[1]}"


def write_table(rows: list[dict]) -> str:
    pct = lambda ms: f"{100 * ms[0]:.1f} ± {100 * ms[1]:.1f}"
    lines = ["| Arm | Seeds | Plain acc % | Harness acc % | Harness gain (pts) | Train tokens | Train s | Infer tokens/ex (plain → harness) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {label(r['arm'])} | {r['seeds']} | {pct(r['plain'])} | {pct(r['harness'])} | {pct(r['gain'])} "
                     f"| {r['train_tokens']:,.0f} | {r['train_seconds']:.0f} | {r['plain_tokens']:.0f} → {r['harness_tokens']:.0f} |")
    table = "\n".join(lines) + "\n"
    (RESULTS / "table.md").write_text(table, encoding="utf-8")
    return table


def plot(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x, w = range(len(rows)), 0.36
    for off, key, color, name in ((-w / 2 - 0.01, "plain", PLAIN, "Plain"), (w / 2 + 0.01, "harness", HARNESS, "Harness")):
        means = [100 * r[key][0] for r in rows]
        errs = [100 * r[key][1] for r in rows]
        bars = ax.bar([i + off for i in x], means, w, color=color, label=name, yerr=errs,
                      error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2}, zorder=2)
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(list(x), [label(r["arm"]) for r in rows], color=INK, fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Exact-match accuracy (%)", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left", ncols=2)
    ax.set_title("Qwen2.5-0.5B on 600 held-out tool calls: harness vs LoRA data recipe",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(RESULTS / "accuracy.png", facecolor=SURFACE)


def main() -> None:
    rows = collect()
    if not rows:
        raise SystemExit("no finished arms under results/")
    print(write_table(rows))
    plot(rows)
    print(f"wrote {RESULTS / 'table.md'} and {RESULTS / 'accuracy.png'}")


if __name__ == "__main__":
    main()
