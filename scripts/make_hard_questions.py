"""
Build our own harder question set (the guidebook invites bringing your own dataset).

Every question needs MORE than one retrieval step, and every gold answer is computed by code from the corpus
(build/events.json, parsed from the infoboxes) or, for the text-only questions, read from the article text and
recorded with the quote it comes from. No gold answer is written by an LLM.

    python scripts/make_hard_questions.py   ->  data/questions/hard_multistep.jsonl
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agr.parse import norm, person_names  # noqa: E402

rng = random.Random(7)
events = json.loads((ROOT / "build" / "events.json").read_text(encoding="utf-8"))
by_id = {e["id"]: e for e in events}


def single_gold(e):
    p = person_names(e["gold"], e["event"])
    return p[0] if len(p) == 1 else None


def single_silver(e):
    p = person_names(e["silver"], e["event"])
    return p[0] if len(p) == 1 else None


def label(e):  # "men's 400 metre freestyle swimming event"
    return f"{e['event'][0].lower() + e['event'][1:]} {e['sport'].lower()} event"


# venue + exact date pairs that identify exactly one event (no ambiguity in the corpus)
pair = Counter((norm(e["venue"]), norm(e["date"])) for e in events if e["venue"] and e["date"])
unique_pair = lambda e: e["venue"] and e["date"] and pair[(norm(e["venue"]), norm(e["date"]))] == 1

Q = []


def add(qtype, question, answer, docs, needs):
    Q.append({"qid": f"hard-{len(Q) + 1:03d}", "qtype": qtype, "question": question, "answer": [answer],
              "gold_doc_ids": docs, "needs": needs})


# 1. two editions back (forces a real hop: Winter Games were 1992 -> 1994 -> 1998, so "YEAR - 8" is wrong)
cands = [e for e in events if e["prev_id"] and by_id[e["prev_id"]]["prev_id"]
         and single_gold(by_id[by_id[e["prev_id"]]["prev_id"]])]
winter = [e for e in cands if e["season"] == "Winter" and e["year"] in (1998, 2002)]
summer = [e for e in cands if e["season"] == "Summer"]
for e in rng.sample(winter, 3) + rng.sample(summer, 3):
    pp = by_id[by_id[e["prev_id"]]["prev_id"]]
    add("two_editions_back",
        f"Who won the gold medal in the {label(e)} at the {e['season']} Olympics held two editions before {e['year']}?",
        single_gold(pp), [e["id"], e["prev_id"], pp["id"]], "find the event at YEAR, step back two editions")

# 2. venue + date -> same event at the previous Games
cands = [e for e in events if unique_pair(e) and e["prev_id"] and single_gold(by_id[e["prev_id"]])
         and e["year"] >= 1996]
for e in rng.sample(cands, 6):
    p = by_id[e["prev_id"]]
    add("venue_then_previous",
        f"The event held at {e['venue']} on {e['date']}: who won the gold medal in that same event at the "
        f"previous Olympics?", single_gold(p), [e["id"], p["id"]],
        "venue + date -> event -> previous edition -> gold")

# 3. how many gold-medal events for an athlete (Athlete -> Event traversal + exact count)
golds = defaultdict(list)
for e in events:
    for n in person_names(e["gold"], e["event"]):
        golds[n].append(e["id"])
medallists = {n for e in events for m in ("gold", "silver", "bronze") for n in person_names(e[m], e["event"])}
sig = lambda n: (norm(n).split()[-1].replace("a", ""), norm(n)[0])        # surname (spelling-tolerant) + initial
sig_count = Counter(sig(n) for n in medallists)
# skip names that a reader could confuse with another medallist (e.g. Gabriella Szabó / Gabriela Szabo)
multi = sorted([n for n, ids in golds.items() if 3 <= len(ids) <= 9 and sig_count[sig(n)] == 1])
for n in rng.sample(multi, 5):
    add("athlete_count", f"According to the provided corpus, in how many Olympic events did {n} win the gold medal?",
        str(len(golds[n])), golds[n], "athlete -> all medal events -> count golds")

# 4. compare two events of the same Games and sport
groups = defaultdict(list)
for e in events:
    if e["competitors"] > 0:
        groups[(e["sport"], e["year"], e["season"])].append(e)
keys = sorted(k for k, v in groups.items() if len(v) >= 4)
for k in rng.sample(keys, 5):
    a, b = rng.sample(groups[k], 2)
    while a["competitors"] == b["competitors"]:
        a, b = rng.sample(groups[k], 2)
    win = a if a["competitors"] > b["competitors"] else b
    add("compare", f"According to the provided corpus, which had more competitors: \"{a['title']}\" or "
                   f"\"{b['title']}\"?", win["title"], [a["id"], b["id"]], "two lookups -> compare")

# 5. the Games held immediately AFTER a year (forward temporal)
cands = [e for e in events if e["next_id"] and single_gold(by_id[e["next_id"]])]
for e in rng.sample(cands, 5):
    n = by_id[e["next_id"]]
    add("next_edition", f"Who won the gold medal in the {label(e)} at the {e['season']} Olympics held immediately "
                        f"after {e['year']}?", single_gold(n), [e["id"], n["id"]], "event at YEAR -> next edition")

# 6. silver medal at a venue + date (medal-role precision)
cands = [e for e in events if unique_pair(e) and single_silver(e)]
for e in rng.sample(cands, 4):
    add("silver_venue", f"Who won the silver medal in the event held at {e['venue']} on {e['date']}?",
        single_silver(e), [e["id"]], "venue + date -> event -> silver (not gold)")

# 7. text-only: articles with no infobox, so the graph has no Event (forces a change of strategy to text search).
#    Gold answers read from the article text; the quote is kept for audit.
TEXT_ONLY = [
    ("Q3400670", "Who won the gold medal in the men's individual pursuit cycling event at the 1992 Summer Olympics?",
     "Chris Boardman", "The Gold medal was won by Briton Chris Boardman"),
    ("Q5198381", "Who won the gold medal in the women's individual pursuit cycling event at the 1992 Summer Olympics?",
     "Petra Rossner", "Gold: | ! | Silver: | ! | Bronze: / Petra Rossner | Kathy Watt | Rebecca Twigg"),
    ("Q3046361", "Who won the gold medal in the women's individual pursuit cycling event at the 2000 Summer Olympics?",
     "Leontien Zijlaard", "Gold: | ! | Silver: | ! | Bronze: / Leontien Zijlaard, NED | Marion Clignet, FRA"),
    ("Q5198385", "Who won the gold medal in the men's individual pursuit cycling event at the 2000 Summer Olympics?",
     "Robert Bartko", "Results table / Robert Bartko, GER | Jens Lehmann, GER | Brad McGee, AUS ... 1 | Robert Bartko"),
    ("Q5198392", "Who won the gold medal in the women's points race cycling event at the 2000 Summer Olympics?",
     "Antonella Bellutti", "Gold: | ! | Silver: | ! | Bronze: / Antonella Bellutti ITA | Leontien Zijlaard NED"),
]
corpus = {json.loads(l)["doc_id"]: json.loads(l)["text"] for l in open(ROOT / "data/corpus/corpus.jsonl", encoding="utf-8")}
for doc, q, a, quote in TEXT_ONLY:
    assert a in corpus[doc], (doc, a)          # the answer really is in that article's text
    add("text_only", q, a, [doc], f"no infobox: graph finds nothing, switch to text search. Source: {quote}")

out = ROOT / "data" / "questions" / "hard_multistep.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for q in Q:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")
print(f"{len(Q)} questions -> {out}", Counter(q["qtype"] for q in Q))
