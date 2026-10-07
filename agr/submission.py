"""Package the hidden-set outputs for submission.

    python -m agr.submission   ->  submission/hidden_<pipeline>.jsonl, submission/hidden_answers.csv, submission/README.md

Each JSONL line is one question with: the answer, its citations (document id, title, Wikipedia URL), confidence,
code-written notes, the grounding verdict, token usage (context / LLM input / LLM output / total), LLM and tool calls,
time, and the full investigation trace (every step: agent, action, arguments, observation, time, tokens).
"""
from __future__ import annotations

import csv
import json

from agr.config import ROOT, SETTINGS
from agr.pipelines import PIPELINES, TOOL_FAMILY

LLM_AGENTS = ("orchestrator", "answer_generator", "evidence_evaluator")
METHOD = {"link": "entity linking", "aggregate": "aggregation (GSQL)", "multi_hop": "multi-hop reasoning",
          "fact_history": "fact history (claims)", "graph": "graph traversal", "vector": "similarity search",
          "document": "document retrieval"}


def guidebook_trace_summary(r: dict) -> dict:
    """The agent-trace items the hackathon guidebook asks for, under their own names, computed from the trace."""
    tools = [s for s in r["steps"] if s["agent"] not in LLM_AGENTS]
    decisions = [s for s in r["steps"] if s["agent"] == "orchestrator"]
    checks = [s for s in r["steps"] if s["agent"] == "evidence_evaluator"]
    return {
        "num_steps": len(r["steps"]),
        "num_retrieval_steps": len(tools),
        "num_reasoning_steps": len(decisions) + len(checks),
        "retrieval_methods_selected": list(dict.fromkeys(METHOD.get(s["action"], METHOD.get(
            TOOL_FAMILY.get(s["action"], ""), s["action"])) for s in tools)),
        "specialised_agents_invoked": list(dict.fromkeys(s["agent"] for s in r["steps"])),
        "tools_called": [s["action"] for s in tools],
        "time_per_operation_ms": [{"step": s["n"], "action": s["action"], "ms": round(s["ms"], 1)} for s in r["steps"]],
        "tokens_per_operation": [{"step": s["n"], "action": s["action"],
                                  "tokens": s["input_tokens"] + s["output_tokens"]} for s in r["steps"]],
        "total_tokens": r["total_tokens"],
        "num_chunks": r.get("chunks_retrieved", 0),
        "num_citations": len(r["citations"]),
        "changed_strategy": r.get("strategy_changes", 0) > 0,
        "stop_reason": r["stop_reason"],
    }

OUT = ROOT / "submission"


def build() -> None:
    OUT.mkdir(exist_ok=True)
    table = {}
    for p in PIPELINES:
        src = ROOT / "results" / f"hidden_{p}.jsonl"
        if not src.exists():
            continue
        rows = sorted({json.loads(l)["qid"]: json.loads(l) for l in open(src, encoding="utf-8")}.values(),
                      key=lambda r: r["qid"])
        with open(OUT / f"hidden_{p}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({
                    "qid": r["qid"], "question": r["question"], "pipeline": p, "model": SETTINGS.model,
                    "answer": r["answer"], "confidence": r.get("confidence", ""),
                    "citations": [{"doc_id": c["doc_id"], "title": c["title"], "url": c.get("url", "")}
                                  for c in r["citations"]],
                    "grounded": r.get("grounded"), "grounding": r.get("grounding", ""), "notes": r.get("notes", []),
                    "tokens": {"context": r["context_tokens"], "llm_input": r["input_tokens"],
                               "llm_output": r["output_tokens"], "total": r["total_tokens"]},
                    "llm_calls": r["llm_calls"], "tool_calls": r["tool_calls"],
                    "active_time_s": round(sum(s["ms"] for s in r["steps"]) / 1000, 2),
                    "stop_reason": r["stop_reason"], "strategy_changes": r.get("strategy_changes", 0),
                    "chunks_retrieved": r.get("chunks_retrieved", 0),
                    "agentic_trace_summary": guidebook_trace_summary(r),
                    "reasoning_chain": r.get("reasoning_chain", []),
                    "trace": [{"step": s["n"], "agent": s["agent"], "action": s["action"], "args": s["args"],
                               "plan": s.get("reasoning", ""), "observation": s["observation"], "found": s["found"],
                               "ms": s["ms"], "llm_input_tokens": s["input_tokens"],
                               "llm_output_tokens": s["output_tokens"], "via": s.get("via", "")}
                              for s in r["steps"]],
                }, ensure_ascii=False) + "\n")
        for r in rows:
            table.setdefault(r["qid"], {"qid": r["qid"], "question": r["question"]})
            table[r["qid"]][f"{p}_answer"] = r["answer"]
            table[r["qid"]][f"{p}_tokens"] = r["total_tokens"]
    cols = ["qid", "question"] + [f"{p}_{k}" for p in PIPELINES for k in ("answer", "tokens")]
    with open(OUT / "hidden_answers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for q in sorted(table):
            w.writerow({c: table[q].get(c, "") for c in cols})
    (OUT / "README.md").write_text(
        "# Hidden-set submission (50 questions)\n\n"
        "* `hidden_agentic.jsonl`: **the submitted system** (Agentic GraphRAG). One JSON object per question: answer, "
        "citations, confidence, code-written notes, grounding verdict, tokens (context / LLM input / LLM output / total), "
        "LLM and tool calls, active time, stop reason, chunks retrieved, the reasoning chain and the full agentic trace "
        "(every step: agent, action, arguments, the orchestrator's stated plan, observation, time, tokens, and whether "
        "the TigerGraph MCP server served it).\n"
        "* `agentic_trace_summary` in each line maps one to one to the guidebook's trace items: number of retrieval and "
        "reasoning steps, retrieval methods selected, specialised agents invoked, tools called, time per operation, "
        "tokens per operation, total tokens, number of chunks and citations, whether the system changed strategy, and "
        "when and why it stopped.\n"
        "* `hidden_graphrag.jsonl`, `hidden_rag.jsonl`: the two baselines, same format.\n"
        "* `hidden_answers.csv`: all three answers and token totals side by side.\n\n"
        f"Model: {SETTINGS.model}. Graph: TigerGraph Savanna, graph `OlympicRAG`.\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
