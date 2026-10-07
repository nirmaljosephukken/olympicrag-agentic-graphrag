"""
Specialised retrieval / reasoning agents. Each one is a deterministic, inspectable tool over TigerGraph that
returns an Observation: a compact summary for the orchestrator plus Evidence items (with citations).

  entity_linker      question text -> years / season / sport / venue / date / exact event titles
  graph_search       filtered traversal Games -> Event (sport, venue, date, event-name filters)
  venue_events       Venue -> Event multi-hop (venue + date -> event -> medallists)
  event_profile      1-hop neighbourhood of an event (medallists, nations, previous/next edition)
  edition_hop        temporal traversal along NEXT_EDITION / PREV_EDITION edges
  aggregate          graph-native count / argmax over events (GSQL accumulators)
  similarity_search  vector search over all chunks (TigerGraph HNSW index)
  scoped_similarity  vector search restricted to chunks of graph-selected documents
  read_document      document retrieval (ordered chunks of one article)
  athlete_medals     Athlete -> Event traversal
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from agr.config import ROOT
from agr.embed import embed_query
from agr.parse import norm, TITLE_RE, split_names
from agr.tg import TG

MONTHS = "january february march april may june july august september october november december".split()


@dataclass
class Evidence:
    id: str
    kind: str          # graph | chunk | aggregate
    doc_id: str
    title: str
    text: str
    url: str = ""


@dataclass
class Observation:
    tool: str
    summary: str                       # what the orchestrator sees
    evidence: list[Evidence] = field(default_factory=list)
    found: bool = True
    data: Any = None
    ms: float = 0.0
    context_tokens: int = 0


def _attrs(v: dict) -> dict:
    return {"id": v.get("v_id"), **v.get("attributes", {})}


def _date_tokens(s: str) -> set[str]:
    s = norm(s)
    toks = set(re.findall(r"\b\d{1,2}\b", s))
    toks |= {m for m in MONTHS if m in s}
    return toks


def date_score(query: str, cand: str) -> float:
    """2.0 = the date text is identical; 1.0+ = one contains the other; else day/month token overlap (0-1)."""
    qn, cn = norm(query).replace(" to ", " "), norm(cand).replace(" to ", " ")
    if not qn or not cn:
        return 0.0
    if qn == cn:
        return 2.0
    q, c = _date_tokens(query), _date_tokens(cand)
    overlap = len(q & c) / len(q | c) if q and c else 0.0
    if qn in cn or cn in qn:
        return 1.0 + overlap / 2
    return overlap


def event_phrase_match(phrase: str, event_norm: str) -> bool:
    p = norm(phrase)
    return bool(p) and re.search(r"(?<![a-z])" + re.escape(p) + r"(?![a-z])", event_norm) is not None


class Toolbox:
    def __init__(self, tg: TG | None = None):
        self.tg = tg or TG()
        self.counter = 0
        self.docs = {d["id"]: d for d in json.loads((ROOT / "build" / "documents.json").read_text(encoding="utf-8"))}

    # ------------------------------------------------------------ helpers
    def _ev(self, kind: str, doc_id: str, text: str, title: str | None = None) -> Evidence:
        self.counter += 1
        d = self.docs.get(doc_id, {})
        return Evidence(id=f"E{self.counter}", kind=kind, doc_id=doc_id, title=title or d.get("title", doc_id),
                        text=text, url=d.get("url", ""))

    def _event_fact(self, e: dict) -> Evidence:
        parts = [f"{e.get('title')}"]
        for k in ("venue", "date", "date_text", "competitors", "nations", "gold", "silver", "bronze"):
            if e.get(k) not in (None, "", -1):
                parts.append(f"{k.replace('_text', '')}: {e[k]}")
        return self._ev("graph", e["id"], "; ".join(parts), e.get("title"))

    # ------------------------------------------------------------ entity linking
    @lru_cache(maxsize=1)
    def catalog(self) -> dict:
        p = ROOT / "build" / "catalog.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        r = self.tg.query("catalog")
        ev = self.tg.query("find_events", {"sport_norm": "", "year": -1, "season": "", "venue_norm": "",
                                           "date_norm": "", "event_norm": "", "lim": 100000})
        cat = {"sports": sorted({s["attributes"]["name"] for s in r[0]["sports"]}),
               "venues": sorted({v["attributes"]["name"] for v in r[0]["venues"]}),
               "events": {norm(e["attributes"]["title"]): e["v_id"] for e in ev[0]["events"]}}
        p.write_text(json.dumps(cat, ensure_ascii=False))
        return cat

    def entity_linker(self, question: str) -> Observation:
        t0 = time.time()
        q = norm(question)
        cat = self.catalog()
        years = [int(y) for y in re.findall(r"\b(19[89]\d|20[0-2]\d)\b", question)]
        season = "Summer" if "summer" in q else ("Winter" if "winter" in q else "")
        sports = sorted([s for s in cat["sports"] if re.search(r"\b" + re.escape(norm(s)) + r"\b", q)],
                        key=len, reverse=True)
        sports = [s for s in sports if not any(norm(s) != norm(o) and norm(s) in norm(o) for o in sports)]
        venues = sorted([v for v in cat["venues"] if len(norm(v)) > 3 and norm(v) in q], key=len, reverse=True)
        titles = [(t, i) for t, i in cat["events"].items() if t in q]
        date = ""
        m = re.search(r"\bon (.+?)(?: at the \d{4}| at the|:|\?|$)", question)
        if m:
            date = m.group(1).strip()
        res = {"years": years, "season": season, "sports": sports[:3], "venues": venues[:3], "date_phrase": date,
               "exact_event_titles": [{"event_id": i, "title_norm": t} for t, i in titles[:3]]}
        return Observation("entity_linker", json.dumps(res, ensure_ascii=False), [], bool(any(res.values())),
                           res, (time.time() - t0) * 1000)

    # ------------------------------------------------------------ graph agents
    def graph_search(self, sport: str = "", year: int = -1, season: str = "", venue: str = "", date: str = "",
                     event: str = "", limit: int = 40) -> Observation:
        t0 = time.time()
        m = TITLE_RE.match(event or "")
        if m:  # a full article title was passed as the event name: use its parts
            sport, year, season, event = sport or m.group("sport"), int(year or -1) if year not in (None, "", -1) \
                else int(m.group("year")), season or m.group("season"), m.group("event")
        r = self.tg.query("find_events", {"sport_norm": norm(sport), "year": int(year or -1), "season": season or "",
                                          "venue_norm": norm(venue), "date_norm": "", "event_norm": norm(event),
                                          "lim": 2000})
        evs = [_attrs(v) for v in r[0]["events"]]
        if event:
            ev_part = lambda e: norm(e["title"].split(" – ")[-1])  # the event name after the dash
            exact = [e for e in evs if ev_part(e) == norm(event)]
            evs = exact or [e for e in evs if event_phrase_match(event, norm(e["title"]))]
        if date:
            evs = sorted(evs, key=lambda e: -date_score(date, e.get("date", "")))
        evs = evs[:limit]
        ev = [self._event_fact(e) for e in evs]
        summ = f"{len(evs)} events" + "".join(f"\n[{x.id}] {e['id']} | {x.text}" for x, e in zip(ev, evs))
        return Observation("graph_search", summ, ev, bool(evs), evs, (time.time() - t0) * 1000)

    def resolve_venues(self, venue: str) -> list[str]:
        """Venue entity resolution: spacing/punctuation-insensitive containment, then token overlap."""
        squash = lambda x: norm(x).replace(" ", "")
        q, qt = squash(venue), set(norm(venue).split())
        names = self.catalog()["venues"]
        hits = [v for v in names if q and (q in squash(v) or (squash(v) in q and len(squash(v)) >= 0.6 * len(q)))]
        if not hits:
            scored = sorted(((len(qt & set(norm(v).split())) / max(1, len(qt | set(norm(v).split()))), v)
                             for v in names), reverse=True)
            hits = [v for sc, v in scored[:3] if sc >= 0.5]
        return hits[:6]

    def venue_events(self, venue: str, date: str = "", year: int = -1) -> Observation:
        t0 = time.time()
        if year in (-1, None, 0):
            ys = re.findall(r"\b(19[89]\d|20[0-2]\d)\b", date or "")
            year = int(ys[0]) if ys else -1
        evs, venues, seen = [], [], set()
        for vn in self.resolve_venues(venue) or [venue]:
            r = self.tg.query("events_at_venue", {"venue_norm": norm(vn), "date_norm": "", "year": int(year)})
            venues += [v["attributes"]["name"] for v in r[0]["venues"]]
            for v in r[1]["events"]:
                if v["v_id"] not in seen:
                    seen.add(v["v_id"]); evs.append(_attrs(v))
        for e in evs:
            e["date_match"] = round(date_score(date, e.get("date", "")), 2) if date else None
        evs.sort(key=lambda e: -(e["date_match"] or 0))
        top = [e for e in evs if not date or e["date_match"] == evs[0]["date_match"]][:8] if evs else []
        ev = [self._event_fact(e) for e in top]
        exact = [e for e in top if (e["date_match"] or 0) >= 2.0]
        note = (f"; EXACT date-text match: {exact[0]['title']} (this is the event asked about)" if len(exact) == 1
                else f"; {len(top)} events share the best date match: compare their date text with the question"
                if len(top) > 1 else "")
        summ = (f"venues matched: {sorted(set(venues))}; {len(evs)} events there{note}; best date matches:" +
                "".join(f"\n[{x.id}] {e['id']} | date_match={e['date_match']} | {x.text}" for x, e in zip(ev, top)))
        return Observation("venue_events", summ, ev, bool(top), top, (time.time() - t0) * 1000)

    def event_profile(self, event_id: str) -> Observation:
        t0 = time.time()
        if not re.fullmatch(r"Q\d+", str(event_id)):  # a title instead of an id: resolve it exactly, never fuzzily
            event_id = self.catalog()["events"].get(norm(str(event_id)), event_id)
        r = self.tg.query("event_profile", {"eid": event_id})
        if not r[0]["event"]:
            return Observation("event_profile", f"no event {event_id}", [], False)
        e = _attrs(r[0]["event"][0])
        meta = r[1]
        x = self._event_fact(e)
        summ = (f"[{x.id}] {e['id']} | {x.text}\nmedallists: {meta['medallists']}\n"
                f"previous_edition: {meta['previous_edition']}\nnext_edition: {meta['next_edition']}")
        x.text += f"; previous edition: {meta['previous_edition']}; next edition: {meta['next_edition']}"
        claims, _ = self._claims(event_id)
        n_conf = sum(c["kind"] == "count_statement" and c["status"].startswith("conflicts") for c in claims)
        n_chg = sum(c["kind"] in ("original_result", "disqualification", "medal_reallocated", "doping_retest")
                    for c in claims)
        if n_conf or n_chg:
            summ += (f"\nVERSIONS: {n_conf} conflicting statement(s), {n_chg} result-change statement(s) in the article; "
                     f"call fact_history('{event_id}') to see them")
        conf = [{"official": f"{e.get('competitors')}/{e.get('nations')}", "text": c["sentence"],
                 "text_value": c["claim_value"]} for c in claims
                if c["kind"] == "count_statement" and c["status"].startswith("conflicts")]
        chg = [{"text": c["sentence"]} for c in claims
               if c["kind"] in ("original_result", "disqualification", "medal_reallocated", "doping_retest")]
        return Observation("event_profile", summ, [x], True, {**e, **meta, "n_conflicts": n_conf, "n_changes": n_chg,
                                                              "conflict_claims": conf, "change_claims": chg},
                           (time.time() - t0) * 1000)

    def edition_hop(self, event_id: str, direction: str = "prev", hops: int = 1) -> Observation:
        t0 = time.time()
        r = self.tg.query("edition_hop", {"eid": event_id, "direction": direction, "hops": int(hops)})
        evs = [_attrs(v) for v in r[0]["events"]]
        ev = [self._event_fact(e) for e in evs]
        summ = (f"{direction} x{hops} from {event_id}: " +
                ("".join(f"\n[{x.id}] {e['id']} | {x.text}" for x, e in zip(ev, evs)) if evs else "no linked edition"))
        return Observation("edition_hop", summ, ev, bool(evs), evs, (time.time() - t0) * 1000)

    def multi_hop(self, path: list | str = "", event_id: str = "", venue: str = "", date: str = "", year: int = -1,
                  sport: str = "", season: str = "", event: str = "") -> Observation:
        """Multi-hop reasoning agent: resolve ONE start event, then walk a planned chain of hops through the graph
        (each hop 'prev' or 'next' along the edition chain), recording the event reached at every hop as evidence.

        The start is an event id/title, or venue + date (+ year), or sport + year (+ season) + event name. If the start
        is ambiguous (several events match equally) it does NOT guess: it returns the candidates and stops, so the
        orchestrator can choose one by id. If a hop has no linked edition, the chain stops there and says so."""
        t0 = time.time()
        hops = [h.strip().lower() for h in (path.replace(">", ",").split(",") if isinstance(path, str) else path or [])
                if str(h).strip()]
        bad = [h for h in hops if h not in ("prev", "next")]
        if bad:
            return Observation("multi_hop", f"unknown hop(s) {bad}: each hop must be 'prev' or 'next'", [], False,
                               {"chain": [], "ties": []}, (time.time() - t0) * 1000)
        # 1. resolve the start event (exact, never fuzzy)
        how, cands = "", []
        if event_id:
            eid = self._resolve_event(event_id)
            prof = self.tg.query("event_profile", {"eid": eid})
            cands = [_attrs(prof[0]["event"][0])] if prof and prof[0]["event"] else []
            how = f"event {eid}"
        elif venue:
            vo = self.venue_events(venue, date, year)
            top = vo.data or []
            exact = [e for e in top if (e.get("date_match") or 0) >= 2.0]
            cands = exact if len(exact) == 1 else top
            how = f"venue '{venue}' + date '{date}'"
        elif sport or event:
            go = self.graph_search(sport=sport, year=year, season=season, event=event, date=date)
            cands = go.data or []
            how = f"graph search {sport} {year} {season} '{event}'".strip()
        if not cands:
            return Observation("multi_hop", f"start not found ({how or 'no start given'})", [], False,
                               {"chain": [], "ties": []}, (time.time() - t0) * 1000)
        if len(cands) > 1:
            ev = [self._event_fact(e) for e in cands[:8]]
            summ = (f"START IS AMBIGUOUS: {len(cands)} events match {how} equally. No hop was taken. Call multi_hop "
                    f"again with the event_id of the one the question means:" +
                    "".join(f"\n[{x.id}] {e['id']} | {x.text}" for x, e in zip(ev, cands)))
            ties = [{"id": e["id"], "title": e["title"], "date": e.get("date", "")} for e in cands[:8]]
            return Observation("multi_hop", summ, ev, False, {"chain": [], "ties": ties}, (time.time() - t0) * 1000)
        # 2. walk the chain, one hop at a time
        cur = cands[0]
        chain, ev = [cur], [self._event_fact(cur)]
        lines = [f"start ({how}): [{ev[0].id}] {cur['id']} | {ev[0].text}"]
        stopped = ""
        for i, h in enumerate(hops, 1):
            r = self.tg.query("edition_hop", {"eid": cur["id"], "direction": h, "hops": 1})
            nxt = [_attrs(v) for v in r[0]["events"]]
            if not nxt:
                stopped = f"hop {i} ({h}) from {cur['title']}: no linked edition, chain stopped"
                lines.append(stopped)
                break
            cur = nxt[0]
            x = self._event_fact(cur)
            chain.append(cur); ev.append(x)
            lines.append(f"hop {i} ({h}): [{x.id}] {cur['id']} | {x.text}")
        done = len(chain) - 1 == len(hops)
        if done:
            lines.append(f"chain complete: answer from the LAST event [{ev[-1].id}]")
        return Observation("multi_hop", "\n".join(lines), ev, done,
                           {"chain": [{"id": e["id"], "title": e["title"]} for e in chain], "ties": [], "how": how,
                            "path": hops, "stopped": stopped}, (time.time() - t0) * 1000)

    def aggregate(self, sport: str, year: int, season: str, threshold: int = -1) -> Observation:
        t0 = time.time()
        r = self.tg.query("aggregate_events", {"sport_norm": norm(sport), "year": int(year), "season": season,
                                               "threshold": int(threshold)})
        s, rows = r[0], [_attrs(v) for v in r[1]["events"]]
        rows.sort(key=lambda e: -e["competitors"])
        self.counter += 1
        table = "; ".join(f"{e['title'].split(' – ')[-1]}={e['competitors']}" for e in rows)
        text = (f"{sport} at the {year} {season} Olympics: {s['events_found']} events in corpus; "
                f"events with competitors > {threshold}: {s['over_threshold']}; max competitors {s['max_competitors']} "
                f"in {s['max_events']}; events without a competitor count: {s['events_without_count']}. Table: {table}")
        ev = Evidence(id=f"E{self.counter}", kind="aggregate", doc_id=",".join(e["id"] for e in rows),
                      title=f"aggregate({sport}, {year} {season})", text=text)
        return Observation("aggregate", f"[{ev.id}] {text}", [ev], s["events_found"] > 0, {**s, "rows": rows},
                           (time.time() - t0) * 1000)

    def athlete_medals(self, name: str) -> Observation:
        """Athlete -> Event traversal. Medal counts are computed in code from the medal fields (exact name match),
        so a count answer is grounded in a stated number rather than the LLM's own tally."""
        t0 = time.time()
        r = self.tg.query("athlete_medals", {"name_like": norm(name)})
        evs = [_attrs(v) for v in r[1]["events"]]
        athletes = [a["attributes"]["name"] for a in r[0]["athletes"]]
        exact = [a for a in athletes if norm(a) == norm(name)] or athletes[:1]
        ev, lines = [], []
        for a in exact:
            by = {"gold": [], "silver": [], "bronze": []}
            for e in evs:
                for medal in by:
                    if any(norm(x) == norm(a) for x in split_names(e.get(medal, ""))):
                        by[medal].append(e)
            for e in by["gold"] + by["silver"] + by["bronze"]:
                ev.append(self._event_fact(e))
            self.counter += 1
            text = (f"{a}: gold in {len(by['gold'])} events; silver in {len(by['silver'])} events; "
                    f"bronze in {len(by['bronze'])} events. Gold: " + "; ".join(e["title"] for e in by["gold"]))
            ev.append(Evidence(id=f"E{self.counter}", kind="aggregate",
                               doc_id=",".join(e["id"] for e in by["gold"]), title=f"medal count({a})", text=text))
            lines.append(f"[E{self.counter}] {text}")
        summ = f"athletes matched: {athletes[:10]}\n" + "\n".join(lines) + \
               "".join(f"\n[{x.id}] {x.doc_id} | {x.text}" for x in ev if x.kind == "graph")
        return Observation("athlete_medals", summ, ev, bool(ev), evs, (time.time() - t0) * 1000)

    # ------------------------------------------------------------ reasoning over time
    def _resolve_event(self, event_id: str) -> str:
        if re.fullmatch(r"Q\d+", str(event_id)):
            return event_id
        return self.catalog()["events"].get(norm(str(event_id)), event_id)

    def _resolve_doc(self, doc_id: str) -> str:
        """A title passed instead of a document id is resolved by exact (normalised) title match, never fuzzily."""
        if re.fullmatch(r"Q\d+", str(doc_id)) or doc_id in self.docs:
            return doc_id
        if not hasattr(self, "_doc_by_title"):
            self._doc_by_title = {norm(d.get("title", "")): i for i, d in self.docs.items()}
        return self._doc_by_title.get(norm(str(doc_id)), doc_id)

    def _claims(self, eid: str) -> tuple[list[dict], list[str]]:
        r = self.tg.query("event_claims", {"eid": eid})
        return [_attrs(c) for c in r[1]["claims"]], r[2]["supersedes"]

    def fact_history(self, event_id: str) -> Observation:
        """All versions of an event's facts (official record, article statements, result changes), which supersedes
        which, and conflicts. The resolution is computed in code: dated changes supersede the original result, the
        infobox is the current official record, and disagreeing statements are reported, never dropped."""
        t0 = time.time()
        eid = self._resolve_event(event_id)
        claims, sup = self._claims(eid)
        title = self.docs.get(eid, {}).get("title", eid)
        if not claims:
            return Observation("fact_history", f"{title}: no conflicting or changed facts recorded; the official "
                                               f"record is the only version", [], True,
                               {"event": eid, "conflicts": [], "changes": []}, (time.time() - t0) * 1000)
        ev, lines, conflicts, changes = [], [], [], []
        record = next((c for c in claims if c["kind"] == "official_record"), None)
        for c in sorted(claims, key=lambda c: (c["kind"] != "official_record", c["id"])):
            kind = "graph" if c["kind"] == "official_record" else "claim"
            x = self._ev(kind, eid, f"[{c['source']}; {c['status']}] {c['sentence']}" +
                         (f" (dates: {c['dates']})" if c["dates"] else ""), title)
            ev.append(x)
            lines.append(f"[{x.id}] {c['kind']}: {x.text}")
            if c["kind"] == "count_statement" and c["status"].startswith("conflicts"):
                conflicts.append({"text": c["sentence"], "official": record["sentence"] if record else ""})
            if c["kind"] in ("original_result", "disqualification", "medal_reallocated", "doping_retest"):
                changes.append({"text": c["sentence"], "dates": c["dates"]})
        res = []
        if conflicts:
            res.append(f"CONFLICT: {len(conflicts)} article statement(s) disagree with the official record; policy: the "
                       f"official record (infobox) is used and the disagreement is reported")
        if changes:
            res.append(f"RESULT CHANGED AFTER THE EVENT: {len(changes)} statement(s); the current result is the "
                       f"official record, the original result is superseded" +
                       (f" ({len(sup)} supersedes edge(s))" if sup else ""))
        summ = f"{title}: " + ("; ".join(res) or "all versions agree") + "".join("\n" + l for l in lines)
        return Observation("fact_history", summ, ev, True, {"event": eid, "title": title, "conflicts": conflicts,
                                                            "changes": changes}, (time.time() - t0) * 1000)

    # ------------------------------------------------------------ text agents
    def _hits(self, r, tool: str, t0: float, max_chars: int) -> Observation:
        dist = r[1]["distances"] if len(r) > 1 else {}
        hits = sorted((_attrs(h) for h in r[0]["hits"]), key=lambda h: dist.get(h["id"], 1))
        ev = [self._ev("chunk", h["doc_id"], h["text"][:max_chars]) for h in hits]
        for x, h in zip(ev, hits):
            x.kind = f"chunk:{h['id']}"
        summ = "".join(f"\n[{x.id}] ({x.title}) {x.text}" for x in ev)
        return Observation(tool, summ.strip(), ev, bool(ev), hits, (time.time() - t0) * 1000,
                           sum(len(x.text) for x in ev) // 4)

    def similarity_search(self, query: str, k: int = 6, max_chars: int = 1400) -> Observation:
        t0 = time.time()
        return self._hits(self.tg.query("chunk_search", {"qv": embed_query(query), "k": int(k)}),
                          "similarity_search", t0, max_chars)

    def scoped_similarity(self, query: str, doc_ids: list[str], k: int = 4, max_chars: int = 1400) -> Observation:
        t0 = time.time()
        return self._hits(self.tg.query("chunk_search_in_docs", {"qv": embed_query(query), "doc_ids": list(doc_ids),
                                                                 "k": int(k)}), "scoped_similarity", t0, max_chars)

    def find_documents(self, query: str, k: int = 5) -> Observation:
        """Document linking: rank article titles by word overlap with the query (code, no LLM, no embeddings).
        Year numbers must match exactly, so '1992' never links to a 2000 article."""
        t0 = time.time()
        qn = set(norm(query).split())
        q_years = {w for w in qn if re.fullmatch(r"(19|20)\d\d", w)}
        scored = []
        for d in self.docs.values():
            tn = set(norm(d["title"]).split())
            t_years = {w for w in tn if re.fullmatch(r"(19|20)\d\d", w)}
            if q_years and t_years and not (q_years & t_years):
                continue
            overlap = len(qn & tn) / max(1, len(tn))
            if overlap > 0:
                scored.append((overlap, d["id"], d["title"]))
        top = sorted(scored, reverse=True)[:k]
        summ = "documents (title match):" + "".join(f"\n  {i} | {t} | match={sc:.2f}" for sc, i, t in top)
        return Observation("find_documents", summ, [], bool(top), [{"id": i, "title": t} for _, i, t in top],
                           (time.time() - t0) * 1000)

    def read_document(self, doc_id: str, max_chunks: int = 2, contains: str = "") -> Observation:
        """Document retrieval: the opening chunks of one article, or (with `contains`) only the chunks of that
        article that contain the keyword, like searching inside the page."""
        t0 = time.time()
        doc_id = self._resolve_doc(doc_id)
        r = self.tg.query("doc_chunks", {"doc_id": doc_id, "max_idx": 500 if contains else int(max_chunks) - 1})
        chunks = [c["attributes"]["text"] for c in r[1]["chunks"]]
        if contains:
            chunks = [c for c in chunks if norm(contains) in norm(c)][:max(2, int(max_chunks))]
        ev = [self._ev("chunk", doc_id, c) for c in chunks]
        summ = "".join(f"\n[{x.id}] {x.text}" for x in ev)
        return Observation("read_document", summ.strip() or f"no matching text in {doc_id}", ev, bool(ev), None,
                           (time.time() - t0) * 1000, sum(len(x.text) for x in ev) // 4)


TOOL_SPECS = {
    "entity_linker": "(question) -> years, season, sport, venue, date phrase and exact event titles found in the text",
    "graph_search": "(sport, year, season, venue, date, event, limit) -> events from the graph matching all given "
                    "filters; 'event' is the event name e.g. \"Men's 20 kilometres walk\" (omit unknown filters)",
    "venue_events": "(venue, date, year) -> events held at a venue, ranked by date match, with gold medallists; "
                    "pass the date text exactly as written in the question",
    "event_profile": "(event_id) -> full facts of one event + medallists + previous/next edition ids",
    "edition_hop": "(event_id, direction='prev'|'next', hops=1) -> the same event at the previous/next Games",
    "multi_hop": "(path=['prev',...], event_id | venue+date+year | sport+year+season+event) -> multi-hop reasoning: "
                 "resolves ONE start event exactly, then walks the planned hops along the edition chain, returning the "
                 "event (with medallists) reached at every hop; reports an ambiguous start instead of guessing",
    "aggregate": "(sport, year, season, threshold) -> exact count of events with competitors > threshold, "
                 "max competitors and which event(s) have it, full per-event table",
    "similarity_search": "(query, k=6) -> top-k text chunks from the whole corpus (vector search)",
    "scoped_similarity": "(query, doc_ids, k=4) -> vector search only inside the given documents",
    "fact_history": "(event_id or title) -> every version of the event's facts: the official record, article statements, "
                    "result changes after the event (disqualifications, reallocations, with dates), what supersedes "
                    "what, and conflicts",
    "find_documents": "(query) -> article titles that best match the query (use when the graph has no event for it)",
    "read_document": "(doc_id, max_chunks=2, contains='') -> opening chunks of one article, or only its chunks that "
                     "contain a keyword such as 'Gold'",
    "athlete_medals": "(name) -> events where an athlete won gold/silver/bronze, with medal counts computed exactly",
}
