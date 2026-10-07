"""
Corpus -> graph entities + text chunks.

The corpus is Wikipedia plain text where most articles start with an infobox block:

    [Infobox Olympic event]
      event: Men's 400 metre freestyle
      games: 2016 Summer
      ...

Olympic infoboxes are parsed deterministically (no LLM extraction cost, no hallucinated edges) into:
  Event, Games, Sport, Venue, Athlete (medal winners), NOC, plus PREV/NEXT edition links.
Every document (Olympic or not: films, people, companies ...) becomes a Document vertex whose
text is split into Chunk vertices for vector search, so nothing in the corpus is unreachable.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path

TITLE_RE = re.compile(r"^(?P<sport>.+?) at the (?P<year>\d{4}) (?P<season>Summer|Winter) Olympics – (?P<event>.+)$")
INFOBOX_RE = re.compile(r"^\[Infobox (?P<kind>[^\]]+)\]\n(?P<body>(?:  .*\n?)+)")
DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")


def norm(s: str) -> str:
    """Lower-case, accent-free, dash-normalised, single-spaced key for matching names."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = s.translate(DASHES).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def to_int(v: str | None) -> int:
    if not v:
        return -1
    m = re.search(r"\d[\d,]*", v)
    return int(m.group(0).replace(",", "")) if m else -1


@dataclass
class Event:
    id: str                 # = doc_id (Wikidata QID)
    title: str
    sport: str
    event: str
    year: int
    season: str
    games: str              # "2016 Summer"
    venue: str = ""
    date: str = ""
    competitors: int = -1
    nations: int = -1
    gold: str = ""
    silver: str = ""
    bronze: str = ""
    gold_noc: str = ""
    silver_noc: str = ""
    bronze_noc: str = ""
    win_value: str = ""
    prev_year: int = -1
    next_year: int = -1
    extra_bronze: list = field(default_factory=list)


def parse_infobox(text: str) -> tuple[str | None, dict, str]:
    m = INFOBOX_RE.match(text)
    if not m:
        return None, {}, text
    fields = {}
    for line in m.group("body").splitlines():
        if ":" in line:
            k, v = line.strip().split(":", 1)
            fields[k.strip()] = v.strip()
    return m.group("kind"), fields, text[m.end():]


OLYMPIC_BLOCK_RE = re.compile(r"\[Infobox Olympic event\]\n((?:  .*\n?)+)")
TOURNAMENT_KINDS = {"international football competition", "international handball competition",
                    "international ice hockey competition", "field hockey", "olympic water polo tournament",
                    "rugby tournament"}


def olympic_infobox(text: str) -> dict | None:
    """The [Infobox Olympic event] block wherever it appears (tennis articles carry it as a second infobox)."""
    m = OLYMPIC_BLOCK_RE.search(text[:6000])
    if not m:
        return None
    fields = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.strip().split(":", 1)
            fields[k.strip()] = v.strip()
    return fields


def tournament_fields(fields: dict) -> dict:
    """Map a team-tournament infobox onto Olympic-event field names. Only fields that exist are copied;
    the champion is a team (usually an NOC code such as 'USA'), never an invented person."""
    pick = lambda *ks: next((fields[k] for k in ks if fields.get(k)), "")
    venue = pick("venue", "venues")
    venue = "" if re.fullmatch(r"\d+", venue) else venue  # football infoboxes put the venue COUNT in 'venues'
    return {"venue": venue, "dates": pick("dates", "date"),
            "competitors": pick("competitors"), "nations": pick("num_teams", "teams", "nations"),
            "gold": pick("champion", "champions", "champion_other", "winners"),
            "silver": pick("second", "second_other", "runnerup"), "bronze": pick("third", "third_other")}


def parse_event(doc: dict, fields: dict) -> Event | None:
    m = TITLE_RE.match(doc["title"])
    games = fields.get("games", "")
    if m:
        sport, year, season, ev = m.group("sport"), int(m.group("year")), m.group("season"), m.group("event")
    else:
        gm = re.match(r"(\d{4}) (Summer|Winter)", games)
        if not gm:
            return None
        year, season = int(gm.group(1)), gm.group(2)
        sport, ev = doc["title"].split(" at the ")[0], fields.get("event", doc["title"])
    return Event(
        id=doc["doc_id"], title=doc["title"], sport=sport, event=ev, year=year, season=season,
        games=f"{year} {season}", venue=fields.get("venue", ""),
        date=fields.get("date") or fields.get("dates", ""),
        competitors=to_int(fields.get("competitors")), nations=to_int(fields.get("nations")),
        gold=fields.get("gold", ""), silver=fields.get("silver", ""), bronze=fields.get("bronze", ""),
        gold_noc=fields.get("goldNOC", ""), silver_noc=fields.get("silverNOC", ""),
        bronze_noc=fields.get("bronzeNOC", ""), win_value=fields.get("win_value", ""),
        prev_year=to_int(fields.get("prev")), next_year=to_int(fields.get("next")),
        extra_bronze=[fields[k] for k in ("bronze2",) if fields.get(k)],
    )


def chunk_text(title: str, text: str, size: int = 1400, overlap: int = 150) -> list[str]:
    """Paragraph-aware chunks; every chunk is prefixed with the article title so it stands alone."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > size:  # very long paragraph/table
            if cur:
                chunks.append(cur); cur = ""
            chunks.append(p[:size]); p = p[size - overlap:]
        if len(cur) + len(p) + 2 > size and cur:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n" if overlap else ""
        cur += ("\n\n" if cur else "") + p
    if cur.strip():
        chunks.append(cur)
    return [f"[{title}]\n{c}" for c in chunks]


def parse_corpus(path: str | Path):
    docs, events, chunks = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            kind, fields, _ = parse_infobox(d["text"])
            docs.append({"id": d["doc_id"], "title": d["title"], "url": d["url"], "kind": kind or "article",
                         "tokens": d.get("approx_tokens", 0)})
            ev = None
            oly = fields if kind == "Olympic event" else olympic_infobox(d["text"])
            if oly:
                ev = parse_event(d, oly)
            elif kind and kind.lower() in TOURNAMENT_KINDS and TITLE_RE.match(d["title"]):
                ev = parse_event(d, tournament_fields(fields))
            if ev:
                events.append(ev)
            for i, c in enumerate(chunk_text(d["title"], d["text"])):
                chunks.append({"id": f"{d['doc_id']}#{i}", "doc_id": d["doc_id"], "idx": i, "text": c})
    link_editions(events)
    return docs, events, chunks


def link_editions(events: list[Event]) -> None:
    """Resolve prev/next infobox years into concrete Event ids (same sport + event name + season)."""
    by_key = {}
    for e in events:
        by_key[(norm(e.sport), norm(e.event), e.season, e.year)] = e
    for e in events:
        e.prev_id = getattr(by_key.get((norm(e.sport), norm(e.event), e.season, e.prev_year)), "id", "")
        e.next_id = getattr(by_key.get((norm(e.sport), norm(e.event), e.season, e.next_year)), "id", "")


def _split_joined(seg: str) -> list[str]:
    """Split names stored run together ('Dani KingLaura Trott', 'Herta AnitașMarioara Trașcă'): a lowercase letter
    directly followed by an uppercase letter that starts a lowercase word. 'McKeon' / 'MacDonald' are kept."""
    parts, start = [], 0
    for i in range(1, len(seg) - 1):
        if seg[i - 1].islower() and seg[i].isupper() and seg[i + 1].islower():
            word_start = seg.rfind(" ", 0, i) + 1
            if seg[word_start:i] in ("Mc", "Mac", "De", "Di", "La", "Le"):
                continue
            parts.append(seg[start:i]); start = i
    parts.append(seg[start:])
    return parts


def split_names(s: str) -> list[str]:
    """Split a medal field into names using only clear separators: commas / semicolons / 'and' / 'cox:', and names
    stored run together. Removes '*' marks, '(h)'-style notes, footnote text and glued NOC codes ('USAIan Crocker')."""
    if not s:
        return []
    s = re.sub(r"\([^)]*\)", " ", s).replace("*", " ")
    s = re.sub(r"Indicates .*$", " ", s)
    s = re.sub(r"(?<=[^\W\d_])and (?=[A-ZÀ-Þ])", ", ", s)      # 'Virtueand Scott' -> 'Virtue, Scott'
    s = re.sub(r"\s*\bcox:\s*", ", ", s)
    out = []
    for seg in re.split(r",|;|\n|\s+and\s+", s):
        for p in _split_joined(seg.strip()):
            p = re.sub(r"^(?:[A-Z]{3}\s?)(?=[A-ZÀ-Þ][^\W\d_]*[a-zà-ÿ])", "", p.strip())  # glued / leading NOC code
            p = " ".join(p.split())
            if p:
                out.append(p)
    return out


CREW_EVENT = re.compile(r"doubles|pairs?\b|team|relay|two|four|eight|quadruple|double sculls|coxless|coxed|tandem|"
                        r"synchron|beach|madison|dance|duet|\b470\b|49er|tornado|\bstar\b|soling|yngling|"
                        r"flying dutchman|tempest|nacra|k-2|k-4|c-2|c-4|2-man|4-man|two-man|four-man|keelboat|"
                        r"sprint\b.*team|mixed", re.I)


def person_names(s: str, event: str = "") -> list[str]:
    """Names that are safe to store as Athlete vertices, or [] when the field cannot be split reliably
    (e.g. 'Viktor Ahn Semion Elistratov Vladimir Grigorev' has no separators). Never guesses a split:
    in crew events (doubles, pairs, relays, boats ...) a part longer than two words is treated as unsplit."""
    parts = split_names(s)
    max_words = 2 if CREW_EVENT.search(event or "") else 4
    max_words = 3 if max_words == 2 and len(parts) > 1 else max_words  # separated crew lists may hold 3-word names
    ok = lambda p: (1 <= len(p.split()) <= max_words and not re.search(r"\d|:", p) and len(p) <= 40
                    and not p.isupper() and p.lower() != "not awarded")
    return parts if parts and all(ok(p) for p in parts) else []



if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "data/corpus/corpus.jsonl"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "build")
    out.mkdir(parents=True, exist_ok=True)
    docs, events, chunks = parse_corpus(src)
    (out / "documents.json").write_text(json.dumps(docs, ensure_ascii=False))
    (out / "events.json").write_text(json.dumps([{**asdict(e), "prev_id": e.prev_id, "next_id": e.next_id}
                                                for e in events], ensure_ascii=False))
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    linked = sum(1 for e in events if e.prev_id or e.next_id)
    print(f"documents={len(docs)} events={len(events)} chunks={len(chunks)} events_with_edition_links={linked}")
