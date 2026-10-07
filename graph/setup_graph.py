"""
Create the OlympicRAG graph on TigerGraph, load documents/events/chunks(+vectors), install queries.

    python -m agr.parse data/corpus/corpus.jsonl build      # parse corpus
    python -m agr.embed build                                # local embeddings -> build/chunk_emb.npy
    python -m graph.setup_graph                              # schema + load + install
Flags: --skip-schema --skip-load --skip-queries
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from agr.config import ROOT
from agr.parse import norm, person_names
from agr.tg import TG

BUILD = ROOT / "build"


def log(*a):
    print("[setup]", *a, flush=True)


def batched(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def create_schema(tg: TG):
    out = tg.gsql("SHOW GRAPH OlympicRAG")
    if "Graph OlympicRAG(" in out and "Chunk" in out:
        log("graph exists, schema step skipped")
        return
    if "Graph OlympicRAG(" not in out:
        log(tg.gsql("CREATE GRAPH OlympicRAG()"))
    log(tg.gsql((ROOT / "graph" / "schema.gsql").read_text())[-800:])


def load(tg: TG):
    docs = json.loads((BUILD / "documents.json").read_text(encoding="utf-8"))
    events = json.loads((BUILD / "events.json").read_text(encoding="utf-8"))
    chunks = [json.loads(l) for l in open(BUILD / "chunks.jsonl", encoding="utf-8")]
    emb = np.load(BUILD / "chunk_emb.npy")
    assert len(emb) == len(chunks), "embeddings out of date: rerun python -m agr.embed"

    for b in batched(docs, 1000):
        tg.upsert({"Doc": {d["id"]: {k: {"value": d[k]} for k in ("title", "url", "kind", "tokens")} for d in b}})
    log(f"docs {len(docs)}")

    V = {"Event": {}, "Games": {}, "Sport": {}, "Venue": {}, "Athlete": {}, "Nation": {}}
    E = {"Doc": {}, "Event": {}, "Games": {}}
    def edge(src_t, src, et, dst_t, dst, attrs=None):
        E.setdefault(src_t, {}).setdefault(src, {}).setdefault(et, {}).setdefault(dst_t, {})[dst] = \
            {k: {"value": v} for k, v in (attrs or {}).items()}

    for e in events:
        V["Event"][e["id"]] = {k: {"value": v} for k, v in {
            "title": e["title"], "title_norm": norm(e["title"]), "sport": e["sport"], "sport_norm": norm(e["sport"]),
            "event": e["event"], "event_norm": norm(e["event"]), "year": e["year"], "season": e["season"],
            "venue": e["venue"], "venue_norm": norm(e["venue"]), "date_text": e["date"], "date_norm": norm(e["date"]),
            "competitors": e["competitors"], "nations": e["nations"], "gold": e["gold"], "silver": e["silver"],
            "bronze": e["bronze"], "gold_noc": e["gold_noc"], "silver_noc": e["silver_noc"],
            "bronze_noc": e["bronze_noc"], "win_value": e["win_value"]}.items()}
        V["Games"][e["games"]] = {"year": {"value": e["year"]}, "season": {"value": e["season"]}}
        V["Sport"][norm(e["sport"])] = {"name": {"value": e["sport"]}}
        edge("Doc", e["id"], "DESCRIBES", "Event", e["id"])
        edge("Event", e["id"], "IN_GAMES", "Games", e["games"])
        edge("Event", e["id"], "OF_SPORT", "Sport", norm(e["sport"]))
        if e["venue"]:
            V["Venue"][norm(e["venue"])] = {"name": {"value": e["venue"]}}
            edge("Event", e["id"], "AT_VENUE", "Venue", norm(e["venue"]))
        for medal in ("gold", "silver", "bronze"):
            names = person_names(e[medal], e["event"])
            if medal == "bronze":
                for extra in e["extra_bronze"]:
                    names += person_names(extra, e["event"])
            for name in names:
                V["Athlete"][norm(name)] = {"name": {"value": name}}
                edge("Event", e["id"], "WON_MEDAL", "Athlete", norm(name), {"medal": medal})
            noc = e[f"{medal}_noc"]
            if noc:
                V["Nation"][noc] = {}
                edge("Event", e["id"], "MEDAL_FOR", "Nation", noc, {"medal": medal})
        if e.get("next_id"):
            edge("Event", e["id"], "NEXT_EDITION", "Event", e["next_id"])
    # Games chain per season
    for season in ("Summer", "Winter"):
        ys = sorted({v["year"]["value"] for k, v in V["Games"].items() if v["season"]["value"] == season})
        for a, b in zip(ys, ys[1:]):
            edge("Games", f"{a} {season}", "NEXT_GAMES", "Games", f"{b} {season}")

    for vt, items in V.items():
        for b in batched(list(items.items()), 2000):
            tg.upsert({vt: dict(b)})
        log(f"{vt} {len(items)}")
    for st, srcs in E.items():
        for b in batched(list(srcs.items()), 1500):
            tg.upsert(edges={st: dict(b)})
    log("edges loaded")

    t0 = time.time()
    for i, b in enumerate(batched(list(range(len(chunks))), 400)):
        verts = {chunks[j]["id"]: {"doc_id": {"value": chunks[j]["doc_id"]}, "idx": {"value": chunks[j]["idx"]},
                                   "text": {"value": chunks[j]["text"]},
                                   "emb": {"value": [round(float(x), 6) for x in emb[j]]}} for j in b}
        eds = {"Doc": {}}
        for j in b:
            eds["Doc"].setdefault(chunks[j]["doc_id"], {"HAS_CHUNK": {"Chunk": {}}})["HAS_CHUNK"]["Chunk"][chunks[j]["id"]] = {}
        tg.upsert(verts and {"Chunk": verts}, eds)
        if i % 10 == 0:
            log(f"chunks {min((i + 1) * 400, len(chunks))}/{len(chunks)} ({time.time() - t0:.0f}s)")
    log("chunks loaded")


def reload_athletes(tg: TG):
    """Delete every Athlete vertex (and its WON_MEDAL edges) and load them again with the current name rules."""
    r = tg._req("DELETE", f"{tg.base}/restpp/graph/{tg.graph}/vertices/Athlete", headers=tg._h(), timeout=600)
    log("deleted athletes:", r.json().get("results"))
    events = json.loads((BUILD / "events.json").read_text(encoding="utf-8"))
    V, E = {}, {}
    for e in events:
        for medal in ("gold", "silver", "bronze"):
            names = person_names(e[medal], e["event"])
            if medal == "bronze":
                for extra in e["extra_bronze"]:
                    names += person_names(extra, e["event"])
            for name in names:
                V[norm(name)] = {"name": {"value": name}}
                E.setdefault(e["id"], {}).setdefault("WON_MEDAL", {}).setdefault("Athlete", {})[norm(name)] = \
                    {"medal": {"value": medal}}
    for b in batched(list(V.items()), 2000):
        tg.upsert({"Athlete": dict(b)})
    for b in batched(list(E.items()), 1500):
        tg.upsert(edges={"Event": dict(b)})
    log(f"athletes {len(V)}")


def build_claims() -> tuple[dict, dict]:
    """Claim vertices + edges for reasoning over time (agr/temporal.py): the infobox record, article statements of the
    totals, and sentences describing a result change. The official record SUPERSEDES the original-result claims."""
    from agr.temporal import event_versions
    events = json.loads((BUILD / "events.json").read_text(encoding="utf-8"))
    corpus = {json.loads(l)["doc_id"]: json.loads(l)["text"]
              for l in open(ROOT / "data/corpus/corpus.jsonl", encoding="utf-8")}
    V, E = {}, {}
    val = lambda **kw: {k: {"value": v} for k, v in kw.items()}
    for e in events:
        v = event_versions(e, corpus.get(e["id"], ""))
        if not (v["counts"] or v["changes"]):
            continue
        rid = f"{e['id']}#record"
        r = v["record"]
        V[rid] = val(event_id=e["id"], kind="official_record", attribute="record", claim_value="",
                     source="infobox (structured, current official record)",
                     sentence=f"gold: {r['gold']}; silver: {r['silver']}; bronze: {r['bronze']}; "
                              f"competitors: {r['competitors']}; nations: {r['nations']}", dates="", status="current")
        E.setdefault(e["id"], {}).setdefault("HAS_CLAIM", {}).setdefault("Claim", {})[rid] = {}
        for i, c in enumerate(v["counts"]):
            cid = f"{e['id']}#count{i}"
            conflict = [x for x in v["count_conflicts"] if x["sentence"] == c["sentence"]]
            V[cid] = val(event_id=e["id"], kind="count_statement", attribute="competitors/nations",
                         claim_value=f"{c['competitors']}/{c['nations']}", source="article text",
                         sentence=c["sentence"], dates="",
                         status="conflicts with the official record" if conflict else "agrees with the official record")
            E[e["id"]]["HAS_CLAIM"]["Claim"][cid] = {}
        for i, c in enumerate(v["changes"]):
            cid = f"{e['id']}#change{i}"
            V[cid] = val(event_id=e["id"], kind=c["kind"], attribute="result", claim_value="", source="article text",
                         sentence=c["sentence"], dates="; ".join(c["dates"]),
                         status="describes a result that was changed later; the current result is the official record")
            E[e["id"]]["HAS_CLAIM"]["Claim"][cid] = {}
            if c["kind"] == "original_result":
                E.setdefault(rid, {}).setdefault("SUPERSEDES", {}).setdefault("Claim", {})[cid] = {}
    return V, E


def load_claims(tg: TG):
    V, E = build_claims()
    for b in batched(list(V.items()), 1500):
        tg.upsert({"Claim": dict(b)})
    ev_edges = {k: v for k, v in E.items() if "#" not in k}
    cl_edges = {k: v for k, v in E.items() if "#" in k}
    for b in batched(list(ev_edges.items()), 1500):
        tg.upsert(edges={"Event": dict(b)})
    for b in batched(list(cl_edges.items()), 1500):
        tg.upsert(edges={"Claim": dict(b)})
    log(f"claims {len(V)}")


def install_queries(tg: TG):
    out = tg.gsql((ROOT / "graph" / "queries.gsql").read_text(), timeout=3600)
    log(out[-1500:])


if __name__ == "__main__":
    tg = TG()
    if "--skip-schema" not in sys.argv:
        create_schema(tg)
    if "--athletes-only" in sys.argv:
        reload_athletes(tg)
    elif "--claims-only" in sys.argv:
        load_claims(tg)
    elif "--skip-load" not in sys.argv:
        load(tg)
        load_claims(tg)
    if "--skip-queries" not in sys.argv:
        install_queries(tg)
    for vt in ("Doc", "Chunk", "Event", "Games", "Venue", "Athlete"):
        log(vt, tg.count(vt))
