"""
Three pipelines over the same TigerGraph instance and the same LLM:

  rag       vector search over all chunks -> answer                               (1 LLM call)
  graphrag  entity linking -> fixed graph retrieval + 1-hop expansion
            + graph-scoped vector search -> answer                                (1 LLM call)
  agentic   orchestrator LLM picks the next specialised agent from the evidence so far, loops,
            an evidence evaluator checks grounding before it is allowed to stop   (2-8 LLM calls)

Every run returns a Result with answer, citations, token accounting and a step-by-step trace.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from agr.config import SETTINGS
from agr.llm import generate, approx_tokens
from agr.parse import norm
from agr import grounding
from agr.tg import take_transports
from agr.tools import Toolbox, Observation, Evidence, TOOL_SPECS


# ----------------------------------------------------------------------------- harness
@dataclass
class Step:
    n: int
    agent: str
    action: str
    args: dict
    observation: str
    found: bool
    ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    context_tokens: int = 0
    reasoning: str = ""
    via: str = ""               # transport that served the graph queries of this step (rest / mcp)


@dataclass
class Result:
    qid: str
    pipeline: str
    question: str
    answer: str
    citations: list[dict]
    confidence: str = ""
    steps: list[Step] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    context_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    latency_ms: float = 0.0
    chunks_retrieved: int = 0
    strategy_changes: int = 0
    stop_reason: str = ""
    grounded: bool = False      # answer fully supported by one cited evidence item (agr/grounding.py)
    grounding: str = ""         # reason / problem from the grounding check
    notes: list[str] = field(default_factory=list)   # code-written notes (e.g. ambiguity), never LLM text
    reasoning_chain: list[str] = field(default_factory=list)  # multi-hop path built by code from tool results

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        return d


def chain_entry(obs: Observation, args: dict) -> str:
    """One link of the multi-hop reasoning chain, written by code from what the tool actually returned."""
    d, t = obs.data, obs.tool
    titles = lambda xs: "; ".join(x.get("title", "")[:70] for x in xs[:3])
    if not obs.found:
        return f"{t}({json.dumps(args, ensure_ascii=False)[:80]}) -> nothing found"
    if t == "entity_linker" and isinstance(d, dict):
        parts = [f"{k}={v}" for k, v in d.items() if v and k != "exact_event_titles"]
        return "linked: " + ", ".join(parts)[:160]
    if t in ("graph_search", "venue_events", "edition_hop", "athlete_medals") and isinstance(d, list):
        head = {"venue_events": f"venue '{args.get('venue', '')}' + date '{args.get('date', '')}'",
                "edition_hop": f"{args.get('direction', 'prev')} edition of {args.get('event_id', '')}",
                "graph_search": "graph search", "athlete_medals": f"athlete '{args.get('name', '')}'"}[t]
        return f"{head} -> {len(d)} event(s): {titles(d)}"
    if t == "event_profile" and isinstance(d, dict):
        return f"event profile -> {d.get('title', '')[:80]}"
    if t == "aggregate" and isinstance(d, dict):
        return (f"aggregate {args.get('sport', '')} {args.get('year', '')} {args.get('season', '')}: "
                f"{d.get('over_threshold')} of {d.get('events_found')} events over {args.get('threshold')}")
    if t == "multi_hop" and isinstance(d, dict):
        if d.get("ties"):
            return f"multi-hop start ambiguous: {len(d['ties'])} candidate events, no hop taken"
        steps = " -> ".join(c["title"][:70] for c in d.get("chain", []))
        return f"multi-hop ({d.get('how', '')}; path {'>'.join(d.get('path', [])) or 'none'}): {steps}" + \
            (f" [{d['stopped']}]" if d.get("stopped") else "")
    if t == "fact_history" and isinstance(d, dict):
        return (f"versions of {d.get('title', '')[:70]}: {len(d.get('conflicts', []))} conflict(s), "
                f"{len(d.get('changes', []))} result change(s)")
    if t == "find_documents" and isinstance(d, list):
        return f"documents -> {titles(d)}"
    if t in ("read_document", "similarity_search", "scoped_similarity"):
        return f"{t} -> {len(obs.evidence)} passage(s) from {', '.join(sorted({e.title[:60] for e in obs.evidence})[:2])}"
    return ""


class Harness:
    """State for one investigation: evidence ledger, token/time accounting, trace."""

    def __init__(self, qid: str, pipeline: str, question: str):
        self.res = Result(qid, pipeline, question, "", [])
        self.ledger: dict[str, Evidence] = {}
        self.ties: list[list[dict]] = []   # groups of events that match the question equally well
        self.versions: dict[str, dict] = {}  # event id -> fact_history data (conflicts, changes)
        self.t0 = time.time()

    def observe(self, agent: str, action: str, args: dict, obs: Observation, reasoning: str = "", llm=None):
        for e in obs.evidence:
            self.ledger[e.id] = e
        if obs.tool == "fact_history" and isinstance(obs.data, dict):
            self.versions[obs.data["event"]] = obs.data
        if obs.tool == "event_profile" and isinstance(obs.data, dict) and \
                (obs.data.get("n_conflicts") or obs.data.get("n_changes")):
            self.versions.setdefault(obs.data.get("id", ""), {"flag": obs.data, "title": obs.data.get("title", "")})
        link = chain_entry(obs, args)
        if link:
            self.res.reasoning_chain.append(link)
        if obs.tool == "venue_events" and isinstance(obs.data, list) and len(obs.data) > 1:
            self.ties.append([{"id": e["id"], "title": e["title"], "date": e.get("date", "")} for e in obs.data])
        if obs.tool == "multi_hop" and isinstance(obs.data, dict) and len(obs.data.get("ties", [])) > 1:
            self.ties.append(obs.data["ties"])
        if obs.tool in ("similarity_search", "scoped_similarity", "read_document"):
            self.res.chunks_retrieved += len(obs.evidence)
        self.res.tool_calls += 1
        ctx = obs.context_tokens or approx_tokens(obs.summary)
        self.res.context_tokens += ctx
        vias = take_transports()
        self.res.steps.append(Step(len(self.res.steps) + 1, agent, action, args, obs.summary[:1500], obs.found,
                                   round(obs.ms, 1), getattr(llm, "input_tokens", 0),
                                   getattr(llm, "output_tokens", 0), ctx, reasoning, ",".join(sorted(set(vias)))))

    def llm(self, agent: str, prompt: str, system: str | None = None, json_mode: bool = False, max_out: int = 1024):
        r = generate(prompt, system=system, json_mode=json_mode, max_output_tokens=max_out)
        self.res.input_tokens += r.input_tokens
        self.res.output_tokens += r.output_tokens
        self.res.llm_calls += 1
        return r

    def log_llm_step(self, agent: str, action: str, r, reasoning: str = "", args: dict | None = None):
        self.res.steps.append(Step(len(self.res.steps) + 1, agent, action, args or {}, r.text[:600], True,
                                   round(r.ms, 1), r.input_tokens, r.output_tokens, 0, reasoning))

    def cite(self, ids: list[str]) -> list[dict]:
        out = []
        for i in ids:
            e = self.ledger.get(str(i).strip("[]"))
            if e:
                out.append({"evidence": e.id, "doc_id": e.doc_id, "title": e.title, "url": e.url, "kind": e.kind,
                            "text": e.text[:1500]})
        return out

    def ground(self, answer: str, ids: list) -> "grounding.Grounding":
        """Run the code grounding check on the cited evidence; add a code-written ambiguity note if the answer's
        event was one of several that matched the question equally (so the ambiguity is reported, not merged)."""
        g = grounding.check(answer, [self.ledger[str(i).strip("[]")] for i in ids if str(i).strip("[]") in self.ledger],
                            self.res.question)
        self.res.grounded, self.res.grounding = g.ok, (g.problem + ": " if g.problem else "") + g.reason
        if g.ok:
            used = next((self.ledger[str(i).strip("[]")] for i in ids if str(i).strip("[]") in self.ledger), None)
            if used:
                self.res.reasoning_chain.append(f"answer '{answer[:80]}' <- {used.id} ({used.title})")
            for eid in g.events:
                v = self.versions.get(eid)
                if not v:
                    continue
                if v.get("conflicts"):
                    self.res.notes.append(f"Conflicting versions in the corpus for {v['title']}: the official record "
                                          f"({v['conflicts'][0]['official']}) and the article text "
                                          f"('{v['conflicts'][0]['text']}') disagree. The answer uses the official record.")
                elif v.get("flag", {}).get("conflict_claims"):
                    c = v["flag"]["conflict_claims"][0]
                    self.res.notes.append(f"Conflicting versions in the corpus for {v['title']}: the official record "
                                          f"(competitors/nations {c['official']}) and the article text "
                                          f"('{c['text']}', i.e. {c['text_value']}) disagree. "
                                          f"The answer uses the official record.")
                if v.get("changes") or v.get("flag", {}).get("change_claims"):
                    v = {**v, "changes": v.get("changes") or v["flag"]["change_claims"]}
                    self.res.notes.append(f"The result of {v['title']} changed after the event "
                                          f"({len(v['changes'])} statement(s), e.g. '{v['changes'][0]['text'][:160]}'). "
                                          f"The official record reflects the current result.")
            for group in self.ties:
                if any(e["id"] in g.events for e in group):
                    chosen = next(e["title"] for e in group if e["id"] in g.events)
                    note = (f"Ambiguous in the corpus: {len(group)} events match this venue and date equally "
                            f"({'; '.join(e['title'] for e in group)}). This answer is for: {chosen}.")
                    if note not in self.res.notes:
                        self.res.notes.append(note)
        return g

    def finish(self) -> Result:
        self.res.latency_ms = round((time.time() - self.t0) * 1000, 1)
        return self.res


ANSWER_RULES = """Answer rules:
- Use ONLY the evidence given. The corpus is the source of truth, even if you remember otherwise.
- Be concise: the answer is a name, a number, or an event title, not a sentence.
- "Which event" questions: answer with the full article title, e.g. "Sailing at the 2000 Summer Olympics – Soling".
- Team medallists: list all names as they appear in the evidence.
- Answer about exactly ONE event. Never combine answers from different events. If several events match the
  question equally well, answer for one of them only and set confidence to "low".
- If the evidence is insufficient, answer "unknown"."""


def _answer_json(h: Harness, agent: str, question: str, context: str) -> dict:
    prompt = (f"{ANSWER_RULES}\n\nEvidence (each item has an id in brackets):\n{context}\n\n"
              f"Question: {question}\n\nReturn JSON: {{\"answer\": str, \"citations\": [evidence ids], "
              f"\"confidence\": \"high|medium|low\"}}")
    r = h.llm(agent, prompt, json_mode=True, max_out=400)
    j = r.json()
    h.log_llm_step(agent, "generate_answer", r)
    return j


# ----------------------------------------------------------------------------- 1. RAG
def run_rag(tb: Toolbox, qid: str, question: str, k: int | None = None) -> Result:
    h = Harness(qid, "rag", question)
    obs = tb.similarity_search(question, k=k or SETTINGS.rag_k)
    h.observe("similarity_search", "chunk_search", {"k": k or SETTINGS.rag_k}, obs)
    j = _answer_json(h, "answer_generator", question, obs.summary)
    h.res.answer, h.res.confidence = str(j.get("answer", "")), j.get("confidence", "")
    h.res.citations = h.cite(j.get("citations", []))
    h.ground(h.res.answer, j.get("citations", []))
    h.res.stop_reason = "single_pass"
    return h.finish()


# ----------------------------------------------------------------------------- 2. GraphRAG
def run_graphrag(tb: Toolbox, qid: str, question: str) -> Result:
    """Fixed retrieval sequence: link -> graph retrieval (+1-hop expansion) -> scoped chunks -> answer."""
    h = Harness(qid, "graphrag", question)
    link = tb.entity_linker(question)
    h.observe("entity_linker", "link", {}, link)
    L = link.data
    year = L["years"][0] if L["years"] else -1
    sport = L["sports"][0] if L["sports"] else ""
    obs_list = []
    if L["exact_event_titles"]:
        for t in L["exact_event_titles"][:2]:
            obs_list.append(("event_profile", {"event_id": t["event_id"]}, tb.event_profile(t["event_id"])))
    elif L["venues"]:
        obs_list.append(("venue_events", {"venue": L["venues"][0], "date": L["date_phrase"], "year": year},
                         tb.venue_events(L["venues"][0], L["date_phrase"], year)))
    elif sport or year > 0:
        o = tb.graph_search(sport=sport, year=year, season=L["season"], limit=40)
        obs_list.append(("graph_search", {"sport": sport, "year": year, "season": L["season"]}, o))
    for name, args, o in obs_list:
        h.observe("graph_retriever", name, args, o)
    # 1-hop expansion: neighbourhood of the (few) retrieved events, as a standard GraphRAG context
    events = [e for _, _, o in obs_list for e in (o.data if isinstance(o.data, list) else [o.data] if o.data else [])]
    if 0 < len(events) <= 3:
        for e in events:
            if "previous_edition" not in e:
                p = tb.event_profile(e["id"])
                h.observe("graph_retriever", "event_profile(1-hop)", {"event_id": e["id"]}, p)
    doc_ids = list({e["id"] for e in events})[:30]
    if doc_ids:
        sc = tb.scoped_similarity(question, doc_ids, k=3)
        h.observe("scoped_similarity", "chunk_search_in_docs", {"docs": len(doc_ids)}, sc)
    else:
        sc = tb.similarity_search(question, k=6)
        h.observe("similarity_search", "chunk_search(fallback)", {"k": 6}, sc)
    context = "\n".join(f"[{e.id}] ({e.title}) {e.text}" for e in h.ledger.values())
    j = _answer_json(h, "answer_generator", question, context)
    h.res.answer, h.res.confidence = str(j.get("answer", "")), j.get("confidence", "")
    h.res.citations = h.cite(j.get("citations", []))
    h.ground(h.res.answer, j.get("citations", []))
    h.res.stop_reason = "single_pass"
    return h.finish()


# ----------------------------------------------------------------------------- 3. Agentic GraphRAG
ORCH_SYSTEM = """You are the orchestrator of an investigation team answering questions over a TigerGraph
knowledge graph built from a Wikipedia corpus (mostly Olympic events 1988-2022, plus films and people).

Graph: Event -IN_GAMES-> Games(year, season); Event -OF_SPORT-> Sport; Event -AT_VENUE-> Venue;
Event -WON_MEDAL-> Athlete; Event -NEXT_EDITION/PREV_EDITION-> Event (same event at adjacent Games);
Doc -HAS_CHUNK-> Chunk(text, vector). Event attributes: competitors, nations, date, venue, gold/silver/bronze.

Specialised agents you can call (one per turn):
{tools}

Each turn, choose the single most useful next action given the question, the linked entities, the evidence
so far and what is still missing. Prefer exact graph tools (cheap, precise) and use text search only when the
graph cannot answer or to verify. Counting and "highest/lowest" questions should use `aggregate`.
Questions that chain several hops ("held immediately before/after", "two Games earlier", "the event held at VENUE on
DATE ... at the previous Games") should use multi_hop: one call resolves the start event exactly (event_id, venue + date,
or sport + year + event) and walks the whole path, e.g. path ["prev"] for "held immediately before YEAR" or ["prev","prev"]
for two editions back. Answer from the LAST event in the chain. If it reports an ambiguous start, pick one event_id and
call it again; use edition_hop for a single hop from an event you already have.
If a call returns nothing, change strategy (relax a filter, other tool) rather than repeating it.
Facts can have several versions. If the question asks who ORIGINALLY won, who holds a medal after a disqualification,
when a result changed, or if an event profile reports VERSIONS, call fact_history: it lists the official record, the
article's statements, what supersedes what and any conflicts. Answer with the version the question asks for (by default
the current official record); conflicts and changes are reported automatically in notes.
Stop as soon as the evidence answers the question: action "final_answer" with the answer and the evidence ids
that support it. Never invent facts that are not in the evidence.

{rules}

Reply with JSON only:
{{"reasoning": "<one short sentence>", "action": "<agent name or final_answer>", "args": {{...}},
  "answer": "<only for final_answer>", "citations": ["E3", ...], "confidence": "high|medium|low"}}"""

TOOL_FAMILY = {"entity_linker": "link", "graph_search": "graph", "venue_events": "graph", "event_profile": "graph",
               "edition_hop": "graph", "aggregate": "graph", "athlete_medals": "graph",
               "similarity_search": "vector", "scoped_similarity": "vector", "read_document": "document",
               "find_documents": "document", "fact_history": "graph", "multi_hop": "graph"}


FEEDBACK = {
    "combined": "The question asks about ONE event, but your answer combines several. Choose the single event whose "
                "details match the question exactly; if several match equally, answer for one of them only "
                "(confidence low). Do not list alternatives in the answer.",
    "wrong_role": "The named athlete is not in the gold field of the cited event. Answer with the gold medallist.",
    "unsupported": "Part of the answer is not in the cited evidence. Answer with the name(s), number or event title "
                   "only (no sentence), exactly as the evidence states it, or gather the missing evidence first.",
    "uncited": "Cite the evidence ids that state the answer.",
    "unknown": "Keep investigating with a different tool or filter before giving up.",
}


def ground_args(action: str, args: dict, linked: dict) -> dict:
    """Argument grounding: keep tool arguments faithful to the question text.
    If the orchestrator shortened the question's date phrase, restore the verbatim phrase the linker found
    (exact date-text matching in venue_events depends on it)."""
    if not isinstance(args, dict):
        return {}
    phrase = (linked or {}).get("date_phrase", "")
    if action in ("venue_events", "graph_search", "multi_hop") and phrase and args.get("date"):
        if norm(args["date"]) and norm(args["date"]) in norm(phrase) and norm(args["date"]) != norm(phrase):
            args = {**args, "date": phrase}
    return args


def run_agentic(tb: Toolbox, qid: str, question: str, max_steps: int | None = None) -> Result:
    max_steps = max_steps or SETTINGS.max_agent_steps
    h = Harness(qid, "agentic", question)
    system = ORCH_SYSTEM.format(tools="\n".join(f"- {k}{v}" for k, v in TOOL_SPECS.items()), rules=ANSWER_RULES)
    # the entity linker always runs first: it is cheap (no LLM) and grounds every later decision
    link = tb.entity_linker(question)
    h.observe("entity_linker", "link", {"question": question}, link)
    history: list[str] = [f"step 1 entity_linker -> {link.summary}"]
    last_family, last_found, rejections = "link", True, 0
    max_rejections = 2
    for step in range(max_steps):
        remaining = max_steps - step
        prompt = (f"Question: {question}\n\nInvestigation so far:\n" + "\n".join(history) +
                  f"\n\nSteps remaining: {remaining}. "
                  + ("You MUST give final_answer now." if remaining == 1 else "Choose the next action."))
        r = h.llm("orchestrator", prompt, system=system, json_mode=True, max_out=500)
        d = r.json()
        action, args = d.get("action", ""), d.get("args") or {}
        if action == "final_answer" and not d.get("answer") and isinstance(args, dict):
            d = {**d, **{k: args[k] for k in ("answer", "citations", "confidence") if k in args}}
        args = ground_args(action, args, link.data)
        h.log_llm_step("orchestrator", f"decide:{action}", r, d.get("reasoning", ""), args)
        if action == "final_answer" or not action:
            ans, cites = str(d.get("answer") or ""), [str(c) for c in (d.get("citations") or [])]
            g = h.ground(ans, cites)
            h.res.steps.append(Step(len(h.res.steps) + 1, "evidence_evaluator", "check_grounding",
                                    {"answer": ans, "citations": cites}, h.res.grounding, g.ok, 0.0))
            if g.ok:
                h.res.answer, h.res.confidence = ans, d.get("confidence", "")
                h.res.citations = h.cite(cites)
                h.res.stop_reason = "evidence_sufficient: " + g.reason
                return h.finish()
            if rejections >= max_rejections or remaining <= 1:
                # never release a rejected answer: withhold it and say why, in code
                h.res.answer, h.res.confidence, h.res.citations = "unknown", "low", []
                h.res.grounded, h.res.grounding = False, "unknown: answer withheld after review"
                h.res.notes.append(f"Answer withheld: the proposed answer failed the grounding check "
                                   f"({g.reason}).")
                for group in h.ties:
                    h.res.notes.append("Candidates that match the question equally: " +
                                       "; ".join(e["title"] for e in group) + ".")
                h.res.stop_reason = f"withheld_after_review: {g.reason}"
                return h.finish()
            rejections += 1
            history.append(f"step {len(history) + 1} evidence_evaluator REJECTED answer '{ans}': {g.reason}. "
                           f"{FEEDBACK.get(g.problem, '')}")
            continue
        fn = getattr(tb, action, None)
        if action not in TOOL_SPECS or fn is None:
            history.append(f"step {len(history) + 1} invalid action '{action}'")
            continue
        try:
            obs = fn(**args)
        except Exception as ex:  # bad args from the LLM -> tell it, keep going
            obs = Observation(action, f"error: {type(ex).__name__}: {str(ex)[:200]}", [], False)
        fam = TOOL_FAMILY.get(action, action)
        if fam != last_family and not last_found:
            h.res.strategy_changes += 1
        last_family, last_found = fam, obs.found
        h.observe(action, action, args, obs, d.get("reasoning", ""))
        summary = obs.summary if len(obs.summary) < 6000 else obs.summary[:6000] + " ...[truncated]"
        history.append(f"step {len(history) + 1} {action}({json.dumps(args, ensure_ascii=False)}) -> {summary}")
    # budget exhausted: answer from what we have
    context = "\n".join(f"[{e.id}] ({e.title}) {e.text}" for e in h.ledger.values())
    j = _answer_json(h, "answer_generator", question, context[:24000])
    h.res.answer, h.res.confidence = str(j.get("answer", "")), j.get("confidence", "")
    h.res.citations = h.cite(j.get("citations", []))
    h.ground(h.res.answer, j.get("citations", []))
    h.res.stop_reason = "step_budget_exhausted"
    return h.finish()


PIPELINES = {"rag": run_rag, "graphrag": run_graphrag, "agentic": run_agentic}
