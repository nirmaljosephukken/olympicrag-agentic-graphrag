"""
Open-corpus question set: questions about the films in the corpus, which are NOT in the graph.

The graph models Olympic events; the corpus also holds 546 film articles. A question about a film that does not name it
can only be answered by finding the article by meaning (similarity search), then reading it. This is the guidebook's
example route "Similarity Search -> Identify Entity -> Retrieve Supporting Documents -> Answer".

Gold answers are taken by code from the film infobox, never written by an LLM. Only films whose director is unique in
the corpus are used, and only fields with a single clean name, so every question has exactly one answer.

  identify   which film (released YEAR, directed by D, starring S)?            gold = the film's name
  composer   who composed the music for the YEAR film directed by D, starring S?  gold = music
  camera     who was the cinematographer of the YEAR film directed by D, starring S?  gold = cinematography

    python scripts/make_open_questions.py  ->  data/questions/open_corpus.jsonl
"""
from __future__ import annotations

import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agr.parse import parse_infobox  # noqa: E402

N_PER_TYPE = 4
clean = lambda v: bool(v) and not re.search(r"[a-z][A-Z]", v) and "," not in v and "(" not in v and len(v) < 40

films = []
for line in open(ROOT / "data/corpus/corpus.jsonl", encoding="utf-8"):
    d = json.loads(line)
    kind, f, _ = parse_infobox(d["text"])
    if kind == "film":
        films.append((d, f))

directors = collections.Counter(f.get("director", "") for _, f in films)
pool = []
for d, f in films:
    year = (re.findall(r"\b(19\d\d|20[0-2]\d)\b", f.get("released", "")) or [""])[0]
    star = (f.get("starring", "").split(",")[0]).strip()
    name = f.get("name") or re.sub(r"\s*\([^)]*\)$", "", d["title"])
    flat = " ".join(d["text"].split())
    if not (year and clean(f.get("director")) and directors[f["director"]] == 1 and clean(star) and clean(name)):
        continue
    if name.lower() in ("the", "it") or name not in flat:
        continue
    pool.append((d, f, year, star, name))

random.seed(2026)
random.shuffle(pool)
Q, used = [], set()


def add(qtype, question, answer, d, source):
    Q.append({"qid": f"open-{len(Q) + 1:03d}", "qtype": qtype, "question": question, "answer": [answer],
              "gold_doc_ids": [d["doc_id"]], "source": source})
    used.add(d["doc_id"])


specs = [("identify", None),
         ("composer", "music"),
         ("camera", "cinematography")]
for qtype, field in specs:
    n = 0
    for d, f, year, star, name in pool:
        if n == N_PER_TYPE:
            break
        if d["doc_id"] in used:
            continue
        who = f"directed by {f['director']} and starring {star}"
        if qtype == "identify":
            add(qtype, f"Which film in the corpus, released in {year}, was {who}?", name, d,
                f"infobox name: {name}; director: {f['director']}; starring: {f['starring'][:80]}")
        else:
            v = f.get(field, "")
            if not clean(v) or v not in " ".join(d["text"].split()):
                continue
            role = "composed the music for" if field == "music" else "was the cinematographer of"
            add(qtype, f"Who {role} the {year} film {who}?", v, d, f"infobox {field}: {v}; director: {f['director']}")
        n += 1

out = ROOT / "data/questions/open_corpus.jsonl"
with open(out, "w", encoding="utf-8") as fh:
    for q in Q:
        fh.write(json.dumps(q, ensure_ascii=False) + "\n")
print(f"{len(Q)} questions -> {out}")
