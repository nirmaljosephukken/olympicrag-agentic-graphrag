"""
Benchmark runner.

    python -m agr.benchmark run   --split public --pipelines rag,graphrag,agentic --workers 4
    python -m agr.benchmark run   --split hidden --pipelines rag,graphrag,agentic
    python -m agr.benchmark judge --split public
    python -m agr.benchmark report

Results are appended to results/<split>_<pipeline>.jsonl (resumable: finished qids are skipped).
Judging: exact/normalised match first (free), then LLM-as-judge (PASS/FAIL + completeness) for the rest.
Grounding: citation precision/recall against the gold documents.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agr.config import ROOT, SETTINGS
from agr.llm import generate
from agr.parse import norm, split_names
from agr.pipelines import PIPELINES
from agr.tools import Toolbox

RES = ROOT / __import__("os").getenv("RESULTS_DIR", "results")  # a second, independent run can go elsewhere
SPLITS = {"public": ROOT / "data/questions/eval_public.jsonl", "hidden": ROOT / "data/questions/eval_hidden.jsonl",
          "hard": ROOT / "data/questions/hard_multistep.jsonl",
          "time": ROOT / "data/questions/time_versions.jsonl",
          "open": ROOT / "data/questions/open_corpus.jsonl"}
_wlock = threading.Lock()


def load_qs(split: str) -> list[dict]:
    return [json.loads(l) for l in open(SPLITS[split], encoding="utf-8")]


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["qid"] for l in open(path, encoding="utf-8") if l.strip()}


def run(split: str, pipelines: list[str], workers: int, limit: int | None):
    RES.mkdir(exist_ok=True)
    qs = load_qs(split)[:limit] if limit else load_qs(split)
    tb = Toolbox()
    tb.catalog()
    for p in pipelines:
        out = RES / f"{split}_{p}.jsonl"
        todo = [q for q in qs if q["qid"] not in done_ids(out)]
        print(f"[{p}] {len(todo)} to run", flush=True)

        def one(q):
            r = PIPELINES[p](tb, q["qid"], q["question"])
            d = r.to_dict()
            d["qtype"] = q.get("qtype")
            with _wlock, open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
            return d

        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(one, q) for q in todo]
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    d = f.result()
                    if i % 10 == 0:
                        print(f"[{p}] {i}/{len(todo)} last={d['qid']} tokens={d['total_tokens']}", flush=True)
                except Exception as e:
                    print(f"[{p}] error: {type(e).__name__}: {e}", flush=True)


# ----------------------------------------------------------------------------- judging
JUDGE = """You grade answers to questions about a document corpus against the ground truth.
Question: {q}
Ground truth: {gold}
System answer: {a}

PASS if the system answer states the same fact as the ground truth (formatting, name order, separators,
accents, or including the full event title vs the short event name do not matter). FAIL if it is wrong,
a different entity/number, "unknown", or hedges between several answers.
completeness: 1.0 if all parts of the ground truth are present (e.g. every team member), else the fraction.
Return JSON: {{"verdict": "PASS"|"FAIL", "completeness": 0.0-1.0, "reason": "<short>"}}"""


def quick_match(answer: str, gold: list[str]) -> bool:
    a = norm(answer)
    for g in gold:
        gn = norm(g)
        if not gn:
            continue
        if a == gn or (re.fullmatch(r"\d+", gn) and re.fullmatch(r"\d+", a or "x") and a == gn):
            return True
        if len(gn) > 6 and gn in a and len(a) < len(gn) * 1.6:
            return True
        if len(a) > 6 and a in gn and len(a) > len(gn) * 0.6:
            return True
        # team answers: same set of people, whatever the separators (gold may store names run together)
        gs = {norm(x) for x in split_names(g)}
        if len(gs) > 1:
            parts = {norm(x) for x in re.split(r",|;|\band\b", re.sub(r"\([^)]*\)", "", answer)) if x.strip()}
            if parts == gs:
                return True
    return False


def _earlier_verdicts(split: str) -> dict:
    """Verdicts already given to the exact same (question, answer) in any earlier judged file, archives included:
    the judge sees only the question, the gold answer and the answer, so an identical pair needs no new LLM call."""
    seen = {}
    for f in list(RES.rglob(f"{split}_*_judged.jsonl")) + list((ROOT / "results").glob(f"{split}_*_judged.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r.get("judge") in ("llm", "llm+rule") and "verdict" in r:
                seen[(r["qid"], r["answer"].strip())] = r
    return seen


def judge(split: str = "public"):
    gold = {q["qid"]: q for q in load_qs(split)}
    earlier = _earlier_verdicts(split)
    for path in sorted(RES.glob(f"{split}_*.jsonl")):
        if path.name.endswith("_judged.jsonl"):
            continue
        out = path.with_name(path.stem + "_judged.jsonl")
        prev = {json.loads(l)["qid"]: json.loads(l) for l in open(out, encoding="utf-8")} if out.exists() else {}
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        rows = list({r["qid"]: r for r in rows}.values())

        def grade(r):
            if r["qid"] in prev and "verdict" in prev[r["qid"]]:
                return prev[r["qid"]]
            g = gold[r["qid"]]
            if quick_match(r["answer"], g["answer"]):
                r.update(verdict="PASS", completeness=1.0, judge="exact", judge_reason="normalised match")
            elif (r["qid"], r["answer"].strip()) in earlier:
                e = earlier[(r["qid"], r["answer"].strip())]
                r.update(verdict=e["verdict"], completeness=e["completeness"], judge=e["judge"],
                         judge_reason=e.get("judge_reason", ""))
            else:
                j = generate(JUDGE.format(q=g["question"], gold=" | ".join(g["answer"]), a=r["answer"]),
                             model=SETTINGS.judge_model, json_mode=True, max_output_tokens=2000).json()
                r.update(verdict=j.get("verdict", "FAIL"), completeness=float(j.get("completeness", 0) or 0),
                         judge="llm", judge_reason=j.get("reason", ""))
                n_gold = max(len(split_names(x)) for x in g["answer"])
                n_ans = len([x for x in re.split(r",|;|\band\b", re.sub(r"\([^)]*\)", "", r["answer"].split(": ")[-1])) if x.strip()])
                if r["verdict"] == "PASS" and n_ans > n_gold and not re.search(r"\d", " ".join(g["answer"])):
                    r.update(verdict="FAIL", judge="llm+rule",
                             judge_reason=f"names {n_ans} candidates where the ground truth has {n_gold} (hedged answer)")
            cited = {d for c in r["citations"] for d in str(c["doc_id"]).split(",")}
            gd = set(g.get("gold_doc_ids", []))
            r["citation_precision"] = round(len(cited & gd) / len(cited), 3) if cited else 0.0
            r["citation_recall"] = round(len(cited & gd) / len(gd), 3) if gd else 0.0
            r["gold"] = g["answer"]
            return r

        with ThreadPoolExecutor(4) as ex:
            graded = list(ex.map(grade, rows))
        with open(out, "w", encoding="utf-8") as f:
            for r in sorted(graded, key=lambda x: x["qid"]):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        acc = sum(r["verdict"] == "PASS" for r in graded) / max(1, len(graded))
        print(f"{path.stem}: accuracy {acc:.1%} over {len(graded)}")


# ----------------------------------------------------------------------------- report
def _mean(xs):
    xs = list(xs)
    return round(statistics.mean(xs), 2) if xs else 0


def _split_metrics(split: str) -> tuple[dict, dict, dict]:
    pipes, byq = {}, {}
    judged = {p: [json.loads(l) for l in open(RES / f"{split}_{p}_judged.jsonl", encoding="utf-8")]
              for p in PIPELINES if (RES / f"{split}_{p}_judged.jsonl").exists()}
    for p, rows in judged.items():
        pipes[p] = {
            "n": len(rows),
            "accuracy": round(sum(r["verdict"] == "PASS" for r in rows) / len(rows), 4),
            "completeness": _mean(r["completeness"] for r in rows),
            "citation_precision": _mean(r["citation_precision"] for r in rows),
            "citation_recall": _mean(r["citation_recall"] for r in rows),
            "avg_total_tokens": _mean(r["total_tokens"] for r in rows),
            "avg_input_tokens": _mean(r["input_tokens"] for r in rows),
            "avg_output_tokens": _mean(r["output_tokens"] for r in rows),
            "avg_context_tokens": _mean(r["context_tokens"] for r in rows),
            "avg_llm_calls": _mean(r["llm_calls"] for r in rows),
            "avg_tool_calls": _mean(r["tool_calls"] for r in rows),
            "avg_latency_s": _mean(_active(r) for r in rows),
            "avg_wall_s": _mean(r["latency_ms"] / 1000 for r in rows),
            "avg_chunks": _mean(r["chunks_retrieved"] for r in rows),
            "total_tokens": sum(r["total_tokens"] for r in rows),
            "tokens_per_correct": round(sum(r["total_tokens"] for r in rows) /
                                        max(1, sum(r["verdict"] == "PASS" for r in rows)), 1),
            "strategy_changes": sum(r.get("strategy_changes", 0) for r in rows),
            "avg_steps": _mean(len(r["steps"]) for r in rows),
        }
        for qt in sorted({r["qtype"] for r in rows}):
            sub = [r for r in rows if r["qtype"] == qt]
            byq.setdefault(qt, {})[p] = {
                "n": len(sub), "accuracy": round(sum(r["verdict"] == "PASS" for r in sub) / len(sub), 4),
                "avg_total_tokens": _mean(r["total_tokens"] for r in sub),
                "avg_llm_calls": _mean(r["llm_calls"] for r in sub),
                "avg_latency_s": _mean(_active(r) for r in sub),
                "avg_citation_recall": _mean(r["citation_recall"] for r in sub)}
    return pipes, byq, judged


def report() -> dict:
    out = {"pipelines": {}, "by_qtype": {}, "per_question": [], "hidden": {}}
    out["pipelines"], out["by_qtype"], judged = _split_metrics("public")
    hp, hq, hard_judged = _split_metrics("hard")
    if hp:
        out["hard"] = {"pipelines": hp, "by_qtype": hq}
        ag = hard_judged.get("agentic", [])
        out["hard"]["agentic_paths"] = _count(" > ".join(s["action"].replace("decide:", "") for s in r["steps"]
                                                         if s["agent"] not in ("orchestrator", "evidence_evaluator"))
                                              for r in ag)
    op, oq, open_judged = _split_metrics("open")
    if op:
        out["open"] = {"pipelines": op, "by_qtype": oq}
        out["open"]["agentic_paths"] = _count(" > ".join(s["action"].replace("decide:", "") for s in r["steps"]
                                                         if s["agent"] not in ("orchestrator", "evidence_evaluator"))
                                              for r in open_judged.get("agentic", []))
    tp, tq, time_judged = _split_metrics("time")
    if tp:
        gold_t = {q["qid"]: q for q in load_qs("time")}
        out["time"] = {"pipelines": tp, "by_qtype": tq, "per_question": []}
        for p, rows in time_judged.items():
            cc = [r for r in rows if gold_t[r["qid"]].get("conflict_values")]
            # a conflict counts as reported only if BOTH values appear in what the user sees (answer + notes)
            rep = [r for r in cc if all(re.search(rf"\b{v}\b", r["answer"] + " " + " ".join(r.get("notes", [])))
                                        for v in gold_t[r["qid"]]["conflict_values"])]
            ch = [r for r in rows if gold_t[r["qid"]]["qtype"] != "count_conflict"]
            tp[p]["conflict_reported"] = f"{len(rep)}/{len(cc)}"
            tp[p]["change_noted"] = f"{sum(any('changed after the event' in n for n in r.get('notes', [])) for r in ch)}/{len(ch)}"
        for qid, g in gold_t.items():
            row = {"qid": qid, "qtype": g["qtype"], "question": g["question"], "gold": g["answer"]}
            for p, rows in time_judged.items():
                r = next((x for x in rows if x["qid"] == qid), None)
                if r:
                    row[p] = {"answer": r["answer"], "verdict": r["verdict"], "tokens": r["total_tokens"],
                              "grounded": r.get("grounded"), "notes": r.get("notes", []),
                              "chain": r.get("reasoning_chain", [])}
            out["time"]["per_question"].append(row)
    # which transport served the graph queries (TigerGraph MCP server or direct RESTPP)
    vias = {}
    for f in RES.glob("*.jsonl"):
        if f.name.endswith("_judged.jsonl"):
            continue
        for l in open(f, encoding="utf-8"):
            for st in json.loads(l).get("steps", []):
                if st.get("via"):
                    k = st["via"].split("(")[0]
                    vias[k] = vias.get(k, 0) + 1
    out["transports"] = vias
    # grounding (hallucination) check on every answer, public and hidden: no gold answers needed
    out["grounding"] = {}
    for split in ("public", "hidden", "hard", "time", "open"):
        for p in PIPELINES:
            f = RES / f"{split}_{p}.jsonl"
            if not f.exists():
                continue
            rows = list({json.loads(l)["qid"]: json.loads(l) for l in open(f, encoding="utf-8")}.values())
            probs = _count((r.get("grounding", "").split(":")[0] if not r.get("grounded") else "supported")
                           for r in rows)
            out["grounding"].setdefault(split, {})[p] = {
                "n": len(rows), "supported": probs.get("supported", 0), "unknown": probs.get("unknown", 0),
                "unsupported": probs.get("unsupported", 0) + probs.get("uncited", 0),
                "combined": probs.get("combined", 0), "wrong_role": probs.get("wrong_role", 0),
                "ambiguity_notes": sum(bool(r.get("notes")) for r in rows)}
    if "agentic" in judged:
        ag = judged["agentic"]
        tools = {}
        for r in ag:
            for s in r["steps"]:
                if s["agent"] not in ("orchestrator", "answer_generator"):
                    t = tools.setdefault(s["agent"], {"calls": 0, "ms": 0.0, "found": 0})
                    t["calls"] += 1; t["ms"] += s["ms"]; t["found"] += bool(s["found"])
        out["agentic_behaviour"] = {
            "tool_usage": {k: {"calls": v["calls"], "avg_ms": round(v["ms"] / v["calls"], 1),
                               "hit_rate": round(v["found"] / v["calls"], 3)} for k, v in tools.items()},
            "avg_steps": _mean(len(r["steps"]) for r in ag),
            "strategy_changes": sum(r["strategy_changes"] for r in ag),
            "questions_with_strategy_change": sum(r["strategy_changes"] > 0 for r in ag),
            "stop_reasons": _count(r["stop_reason"].split(":")[0] for r in ag),
            "paths": _count(" > ".join(s["action"].replace("decide:", "") for s in r["steps"]
                                       if s["agent"] not in ("orchestrator", "evidence_evaluator"))
                            for r in ag),
        }
    by_q = {}
    for p, rows in judged.items():
        for r in rows:
            q = by_q.setdefault(r["qid"], {"qid": r["qid"], "qtype": r["qtype"], "question": r["question"],
                                           "gold": r["gold"]})
            q[p] = {"answer": r["answer"], "verdict": r["verdict"], "tokens": r["total_tokens"],
                    "llm_calls": r["llm_calls"], "latency_s": round(_active(r), 2),
                    "tools": r["tool_calls"], "stop": r["stop_reason"],
                    "grounded": r.get("grounded"), "grounding": r.get("grounding", ""), "notes": r.get("notes", []),
                    "citations": [c["title"] for c in r["citations"]][:4]}
            if p == "agentic":
                q["trace"] = [{"n": s["n"], "agent": s["agent"], "action": s["action"], "args": s["args"],
                               "why": s["reasoning"], "ms": s["ms"], "tok": s["input_tokens"] + s["output_tokens"],
                               "found": s["found"], "obs": s["observation"][:280]} for s in r["steps"]]
    out["per_question"] = sorted(by_q.values(), key=lambda x: x["qid"])
    for p in PIPELINES:
        f = RES / f"hidden_{p}.jsonl"
        if f.exists():
            rows = [json.loads(l) for l in open(f, encoding="utf-8")]
            out["hidden"][p] = {"n": len(rows), "avg_total_tokens": _mean(r["total_tokens"] for r in rows),
                                "unknown_answers": sum(norm(r["answer"]) in ("", "unknown") for r in rows)}
    (RES / "metrics.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for p, m in out["pipelines"].items():
        print(f"{p:9} acc={m['accuracy']:.1%} tokens/q={m['avg_total_tokens']} llm/q={m['avg_llm_calls']} "
              f"lat={m['avg_latency_s']}s tokens/correct={m['tokens_per_correct']}")
    return out


def _active(r) -> float:
    """Active latency (s): tool + LLM time, excluding free-tier rate-limit waits."""
    return sum(s["ms"] for s in r["steps"]) / 1000


def _count(xs):
    d = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return dict(sorted(d.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "judge", "report"])
    ap.add_argument("--split", default="public")
    ap.add_argument("--pipelines", default="rag,graphrag,agentic")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.split, a.pipelines.split(","), a.workers, a.limit)
    elif a.cmd == "judge":
        judge(a.split)
    else:
        report()
