"""Ask one question and watch the investigation.

    python -m agr.ask "question"                 # agentic (default), prints the trace
    python -m agr.ask "question" --all           # run all three pipelines side by side
"""
from __future__ import annotations

import argparse
import json

from agr.pipelines import PIPELINES
from agr.tools import Toolbox


def show(r) -> None:
    print(f"\n=== {r.pipeline.upper()} ===")
    for s in r.steps:
        meta = f"{s.ms:.0f} ms" + (f", {s.input_tokens + s.output_tokens} tok" if s.input_tokens else "") \
            + (f", via {s.via}" if getattr(s, "via", "") else "")
        print(f"[{s.n}] {s.agent:<18} {s.action:<22} ({meta})")
        if s.reasoning:
            print(f"      why: {s.reasoning}")
        if s.args and s.agent != "orchestrator":
            print(f"      args: {json.dumps(s.args, ensure_ascii=False)}")
        print("      " + s.observation[:300].replace("\n", "\n      "))
    print(f"\nANSWER: {r.answer}   (confidence {r.confidence})")
    for c in r.citations:
        print(f"  cites {c['evidence']}: {c['title']}  {c['url']}")
    for n in getattr(r, "notes", []) or []:
        print(f"  NOTE: {n}")
    if getattr(r, "reasoning_chain", None):
        print("reasoning chain:")
        for c in r.reasoning_chain:
            print(f"  {c}")
    print(f"tokens: {r.total_tokens} (in {r.input_tokens}, out {r.output_tokens}, context {r.context_tokens}) · "
          f"LLM calls {r.llm_calls} · tool calls {r.tool_calls} · stop: {r.stop_reason}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--pipeline", default="agentic", choices=list(PIPELINES))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    tb = Toolbox()
    for p in (PIPELINES if a.all else [a.pipeline]):
        show(PIPELINES[p](tb, "cli", a.question))
