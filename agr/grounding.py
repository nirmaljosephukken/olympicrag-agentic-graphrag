"""
Grounding check: is an answer fully supported by ONE piece of cited evidence?

Lesson from our previous TigerGraph submission: an LLM-written explanation merged two different link mechanisms
into one claim that the data did not support. Here the same failure would be an answer that stitches together
facts from different events, names a silver medallist as the gold, or contains a name that no cited evidence holds.
So the check is done in code, not by the LLM, and it is strict:

  1. every name / number in the answer must appear in the cited evidence (no unsupported content);
  2. a non-numeric answer must be supported by a SINGLE evidence item, i.e. one event (no combining events);
  3. for "who won the gold medal" questions on structured evidence, the names must sit in that event's
     `gold:` field (no medal-role confusion).

The agent's evidence evaluator uses it before the agent may stop; the benchmark report runs the same check on
every pipeline's answers to measure unsupported and combined answers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from agr.parse import norm, TITLE_RE


@dataclass
class Grounding:
    ok: bool
    reason: str
    supporting: list[str] = field(default_factory=list)   # evidence ids that hold the full answer
    events: list[str] = field(default_factory=list)       # distinct documents the answer draws on
    problem: str = ""                                      # unknown | uncited | unsupported | combined | wrong_role


def _squash(s: str) -> str:
    return norm(s).replace(" ", "")


def answer_items(answer: str) -> list[str]:
    """Split an answer into the claims it makes: 'A, B and C' -> [A, B, C]; strips '(...)' and 'Label: ' parts."""
    parts = []
    for seg in re.split(r";|\n", answer):
        seg = re.sub(r"\([^)]*\)", "", seg)
        if ": " in seg:
            seg = seg.split(": ")[-1]
        parts += [x.strip(" .") for x in re.split(r",|\band\b|&|/", seg) if x.strip(" .")]
    return parts or [answer.strip()]


def _contains(text: str, item: str) -> bool:
    t, i = norm(text), norm(item)
    if not i:
        return False
    if re.search(r"(?<![a-z0-9])" + re.escape(i) + r"(?![a-z0-9])", t):
        return True
    return len(_squash(item)) >= 6 and _squash(item) in _squash(text)  # team names stored run together


MEDAL_TABLE = re.compile(r"Gold:[\s|!]*Silver:[\s|!]*Bronze:[\s/]*([^|/\n]+)\|([^|/\n]+)\|([^|/\n]+)", re.I)


def _gold_field(text: str) -> str | None:
    """The gold medallist(s) as stated in the evidence, or None when the evidence has no medal field.
    Handles structured facts ('gold: X; silver: Y') and article medal tables ('Gold: | Silver: | Bronze: / X | Y | Z'),
    where the gold winner is the FIRST cell, not everything after the word 'Gold'."""
    t = MEDAL_TABLE.search(text)
    if t:
        return t.group(1)
    m = re.search(r"\bgold:\s*([^;\n|]*)", text)
    return m.group(1) if m and m.group(1).strip() else None


def check(answer: str, cited: list, question: str = "") -> Grounding:
    """`cited` is a list of objects/dicts with id, doc_id, title, text, kind."""
    get = lambda e, k: (e.get(k) if isinstance(e, dict) else getattr(e, k, "")) or ""
    a = (answer or "").strip()
    if norm(a) in ("", "unknown", "none", "n a"):
        return Grounding(False, "answer is unknown", problem="unknown")
    if not cited:
        return Grounding(False, "no evidence cited", problem="uncited")
    texts = {get(e, "id"): get(e, "title") + " | " + get(e, "text") for e in cited}
    docs = {get(e, "id"): get(e, "doc_id") for e in cited}
    kinds = {get(e, "id"): get(e, "kind") for e in cited}

    # numbers: must equal the number stated in the right field of the cited evidence
    if re.fullmatch(r"\d[\d,]*", a):
        n = a.replace(",", "")
        field_re = {"aggregate": r"(?:competitors > \d+: |(?:gold|silver|bronze) in )" + n + r"\b",
                    "graph": r"(?:competitors|nations): " + n + r"\b"}
        sup = [i for i, t in texts.items()
               if re.search(field_re.get(kinds[i], r"(?<![\d.,])" + n + r"(?![\d.,])"), t)]
        return (Grounding(True, "number stated in cited evidence", sup, sorted({docs[i] for i in sup}))
                if sup else Grounding(False, f"the number {a} is not stated in the cited evidence",
                                      problem="unsupported"))

    # an answer that names two or more different cited events is combining them, whatever its wording
    named = {get(e, "doc_id") for e in cited if get(e, "title") and _contains(a, get(e, "title"))}
    if len(named) >= 2:
        return Grounding(False, f"the answer combines {len(named)} different events; the question asks about one",
                         events=sorted(named), problem="combined")

    # whole answer (e.g. an event title) in one item
    whole = [i for i, t in texts.items() if _contains(t, a)]
    items = answer_items(a)
    per_item = {it: [i for i, t in texts.items() if _contains(t, it)] for it in items}
    missing = [it for it, sup in per_item.items() if not sup]
    if missing and not whole:
        return Grounding(False, f"not in the cited evidence: {', '.join(missing)}", problem="unsupported")
    single = whole or [i for i in texts if all(i in sup for sup in per_item.values())]
    if not single:
        used = sorted({docs[i] for sup in per_item.values() for i in sup})
        titles = sorted({get(e, "title") for e in cited if get(e, "id") in {i for s in per_item.values() for i in s}})
        return Grounding(False, f"the answer combines facts from {len(titles)} different events/sources "
                                f"({'; '.join(titles)[:300]}); the question asks about one",
                         events=used, problem="combined")
    # medal role on structured evidence
    asks_original = re.search(r"\boriginal(ly)?\b|\binitially\b|\bfirst (?:awarded|declared)\b|\bstripped\b|"
                              r"\bbefore (?:the |his |her |their )?(?:disqualif|result|change)", question, re.I)
    if "gold medal" in question.lower() and not TITLE_RE.match(a) and not asks_original:
        role_ok = []
        for i in single:
            g = _gold_field(texts[i])
            if g is None or _contains(g, a) or all(_contains(g, it) for it in items):
                role_ok.append(i)
        if not role_ok:
            return Grounding(False, "the named athlete(s) are not the gold medallist(s) in the cited evidence",
                             problem="wrong_role")
        single = role_ok
    return Grounding(True, "answer fully supported by one cited evidence item", single,
                     sorted({docs[i] for i in single}))
