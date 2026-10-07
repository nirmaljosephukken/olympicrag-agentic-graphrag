"""
Reproducibility check: compare two complete benchmark runs question by question.

    python scripts/compare_runs.py results results_confirm  ->  results_confirm/comparison.md

For every split and pipeline it reports accuracy, tokens and grounding in each run, how many questions got the
same verdict in both, and lists every question whose verdict changed, so a difference can be read in the traces.
The hidden set has no gold answers, so it is compared on answers (normalised) and grounding.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agr.parse import norm  # noqa: E402

A, B = (ROOT / a for a in (sys.argv[1:3] if len(sys.argv) > 2 else ("results", "results_confirm")))
NAMES = {"rag": "RAG", "graphrag": "GraphRAG", "agentic": "Agentic GraphRAG"}


def load(d: Path, split: str, p: str, judged: bool) -> dict:
    f = d / f"{split}_{p}{'_judged' if judged else ''}.jsonl"
    return {json.loads(l)["qid"]: json.loads(l) for l in open(f, encoding="utf-8")} if f.exists() else {}


out = [f"# Reproducibility: `{A.name}` vs `{B.name}`", "",
       "Two complete, independent runs of all three pipelines on every question set, same code, same graph, same LLM.", ""]
changed = []
for split in ("public", "hard", "time", "open"):
    out += [f"## {split}", "", "| Pipeline | Accuracy run 1 | Accuracy run 2 | Same verdict | Tokens/q run 1 | Tokens/q run 2 | "
            "Grounded run 1 | Grounded run 2 |", "|---|---|---|---|---|---|---|---|"]
    for p in NAMES:
        a, b = load(A, split, p, True), load(B, split, p, True)
        if not a or not b:
            continue
        common = sorted(set(a) & set(b))
        acc = lambda d: sum(d[q]["verdict"] == "PASS" for q in common) / len(common)
        tok = lambda d: sum(d[q]["total_tokens"] for q in common) / len(common)
        grd = lambda d: sum(bool(d[q].get("grounded")) for q in common)
        same = sum(a[q]["verdict"] == b[q]["verdict"] for q in common)
        out.append(f"| {NAMES[p]} | {acc(a):.0%} | {acc(b):.0%} | {same}/{len(common)} | {tok(a):,.0f} | {tok(b):,.0f} | "
                   f"{grd(a)}/{len(common)} | {grd(b)}/{len(common)} |")
        changed += [(split, p, q, a[q]["answer"], a[q]["verdict"], b[q]["answer"], b[q]["verdict"])
                    for q in common if a[q]["verdict"] != b[q]["verdict"]]
    out.append("")

out += ["## hidden (no gold answers: compared on the answer itself)", "",
        "| Pipeline | Same answer | Unknown run 1 | Unknown run 2 | Grounded run 1 | Grounded run 2 |", "|---|---|---|---|---|---|"]
for p in NAMES:
    a, b = load(A, "hidden", p, False), load(B, "hidden", p, False)
    if not a or not b:
        continue
    common = sorted(set(a) & set(b))
    same = sum(norm(a[q]["answer"]) == norm(b[q]["answer"]) for q in common)
    unk = lambda d: sum(norm(d[q]["answer"]) in ("", "unknown") for q in common)
    grd = lambda d: sum(bool(d[q].get("grounded")) for q in common)
    out.append(f"| {NAMES[p]} | {same}/{len(common)} | {unk(a)} | {unk(b)} | {grd(a)}/{len(common)} | {grd(b)}/{len(common)} |")

out += ["", "## Questions whose verdict changed between the runs", ""]
if changed:
    out += ["| Set | Pipeline | Question | Run 1 | Run 2 |", "|---|---|---|---|---|"]
    out += [f"| {s} | {NAMES[p]} | {q} | {ra[:60]} ({va}) | {rb[:60]} ({vb}) |" for s, p, q, ra, va, rb, vb in changed]
else:
    out.append("None.")
(B / "comparison.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
