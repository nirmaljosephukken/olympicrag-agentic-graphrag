"""
Reasoning-over-time question set: facts with several versions in the corpus.

Hand-curated from sentences found by agr/temporal.py, then verified by code below: every gold answer must appear in
the infobox field or the quoted article sentence it is taken from. No gold answer is written by an LLM.

  original      who won before the result was changed (article text: "X originally won ...")
  current       who is credited after the change (infobox = current official record)
  when          the year a dated decision changed the result (article text)
  vacant        the official record shows the medal as vacant (withdrawn, not reallocated)
  count_conflict  the infobox and the article text give different totals; expected answer = the official record,
                  and the conflict should be REPORTED (measured separately as `conflict_reported`)

    python scripts/make_time_questions.py  ->  data/questions/time_versions.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agr.temporal import event_versions  # noqa: E402

events = {e["id"]: e for e in json.loads((ROOT / "build" / "events.json").read_text(encoding="utf-8"))}
corpus = {json.loads(l)["doc_id"]: json.loads(l)["text"] for l in open(ROOT / "data/corpus/corpus.jsonl", encoding="utf-8")}
T = lambda i: events[i]["title"]

ORIGINAL = [  # (event id, original winner, quoted sentence)
    ("Q25239321", "Mitko Grablev", "Mitko Grablev originally won this category, but he was disqualified"),
    ("Q25239533", "Angel Genchev", "Angel Genchev originally won this category but he was disqualified"),
    ("Q2311975", "Adrián Annus", "Adrián Annus of Hungary originally won the competition, but he was disqualified"),
    ("Q781406", "Johann Mühlegg", "Johann Mühlegg of Spain originally won the competition, but failed the doping test"),
    ("Q26212147", "Nijat Rahimov", "Nijat Rahimov originally won the gold medal but was disqualified in March 2022"),
]
CURRENT = ["Q25239321", "Q25239533", "Q2311975", "Q781406", "Q376518"]  # gold = infobox gold now
WHEN = [
    ("Q26212147", "2022", "In which year was Nijat Rahimov disqualified from his gold medal result in {t}?",
     "disqualified in March 2022 by the Court of Arbitration for Sport"),
    ("Q241789", "2016", "In which year was it announced that Cristina Iovu had failed her re-tested sample, changing "
                        "the result of {t}?", "on 27 July 2016, the International Weightlifting Federation announced"),
]
COUNT = ["Q743905", "Q62020023", "Q2036655", "Q26215191", "Q252875"]

Q = []


def add(qtype, q, a, docs, **kw):
    Q.append({"qid": f"time-{len(Q) + 1:03d}", "qtype": qtype, "question": q, "answer": [a], "gold_doc_ids": docs, **kw})


for i, name, quote in ORIGINAL:
    assert quote in " ".join(corpus[i].split()) and name in quote, (i, name)
    add("original", f"Who originally won the gold medal in {T(i)}, before the result was changed?", name, [i],
        source=quote)
for i in CURRENT:
    v = event_versions(events[i], corpus[i])
    assert v["changes"], i                      # the article really describes a change
    add("current", f"After the result was changed, who is credited with the gold medal in {T(i)}?",
        events[i]["gold"], [i], source=f"infobox gold: {events[i]['gold']}")
for i, year, q, quote in WHEN:
    assert quote in " ".join(corpus[i].split()) and year in quote, (i, year)
    add("when", q.format(t=T(i)), year, [i], source=quote)
assert events["Q26212147"]["gold"] == "vacant"
add("vacant", f"According to the official record in the corpus, who holds the gold medal in {T('Q26212147')}?",
    "vacant", ["Q26212147"], source="infobox gold: vacant; 'As of March 2022, medals for this event have not been "
                                    "reallocated'")
for i in COUNT:
    v = event_versions(events[i], corpus[i])
    conf = [c for c in v["count_conflicts"] if c["attribute"] == "competitors"]
    assert conf, i                               # the conflict really exists
    add("count_conflict", f"How many competitors took part in {T(i)}?", str(events[i]["competitors"]), [i],
        source=f"infobox competitors: {events[i]['competitors']}; article text: '{conf[0]['sentence']}'",
        conflict_values=[str(events[i]["competitors"]), str(conf[0]["text_value"])])

out = ROOT / "data" / "questions" / "time_versions.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for q in Q:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")
print(f"{len(Q)} questions -> {out}")
