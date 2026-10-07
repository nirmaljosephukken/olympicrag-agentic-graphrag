"""
Reasoning over time: versions of a fact, which one supersedes which, and how sure we are.

Two kinds of fact versions exist in this corpus, both extracted by code (no LLM):

1. Result changes. 252 Olympic event articles describe a result that changed after the event: a medallist "originally
   won" and was later disqualified / stripped / re-tested, and medals were reallocated. The infobox holds the CURRENT
   official result; the article text holds the ORIGINAL result and the dated decision that changed it.
2. Count conflicts. The infobox states competitors / nations; the article text often restates them ("Forty-two
   athletes from 32 nations competed"). When the two disagree, that is a conflicting version of the same fact.

Source policy (what supersedes what):
  * a dated decision in the text (disqualification, stripping, reallocation) supersedes the original result;
  * the infobox is the structured current record, so for "who holds the medal now" it is authoritative;
  * for counts, the infobox is used as the official figure and any disagreeing text is reported as a conflict with the
    exact sentence, never silently dropped; units that are not individuals (pairs, teams, boats, crews) are not compared.
"""
from __future__ import annotations

import re

UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
         "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
NUMBER_WORDS = r"(?:(?:one|two|three|four|five|six|seven|eight|nine)\s+hundred(?:\s+and)?\s+)?" \
               r"(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[-\s](?:one|two|three|four|five|six|" \
               r"seven|eight|nine))?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|" \
               r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)"
NUM = rf"(\d{{1,3}}|{NUMBER_WORDS})"
PEOPLE = r"athletes|competitors|swimmers|cyclists|rowers|boxers|fencers|wrestlers|shooters|skaters|skiers|sailors|" \
         r"judokas|gymnasts|riders|archers|divers|weightlifters|lifters|biathletes|players|runners|jumpers|throwers|" \
         r"walkers|canoeists|kayakers|triathletes|lugers|snowboarders|climbers|golfers|equestrians|pentathletes"
COUNT_RE = re.compile(rf"\b{NUM}\s+({PEOPLE})\s+from\s+{NUM}\s+(?:nations|countries|NOCs)\s+"
                      r"(?:competed|took part|participated|entered|competing|taking part|participating|were entered)",
                      re.I)
CHANGE_KEY = r"stripped of|disqualified|reallocated|re-allocated|upgraded to|later awarded|retroactively|annulled|" \
             r"originally (?:won|finished|placed|awarded|took)|initially (?:won|finished|placed)|retest|re-test|reanaly"
CHANGE_RE = re.compile(r"[^.\n]*\b(?:" + CHANGE_KEY + r")[^.\n]*\.", re.I)
DATE_RE = re.compile(r"\b(?:(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|September|October|"
                     r"November|December)\s+(?:\d{1,2},\s+)?)?(?:19[89]\d|20[0-2]\d)\b")


def words_to_int(s: str) -> int | None:
    s = s.lower().replace("-", " ").replace(" and ", " ").strip()
    if s.isdigit():
        return int(s)
    total, cur = 0, 0
    for w in s.split():
        if w in UNITS:
            cur += UNITS[w]
        elif w in TENS:
            cur += TENS[w]
        elif w == "hundred":
            cur = max(cur, 1) * 100
        else:
            return None
    return total + cur


def count_claims(text: str) -> list[dict]:
    """'Forty-two athletes from 32 nations competed' -> {competitors: 42, nations: 32, sentence}."""
    out = []
    for m in COUNT_RE.finditer(text):
        if re.match(r"\s*in the (?:final|semi|heat|quarter|round|qualif|repechage|medal)", text[m.end():m.end() + 30], re.I):
            continue  # a count for one round, not for the whole event
        a, b = words_to_int(m.group(1)), words_to_int(m.group(3))
        if a is None or b is None:
            continue
        start = text.rfind(".", 0, m.start()) + 1
        end = text.find(".", m.end())
        out.append({"competitors": a, "nations": b, "unit": m.group(2).lower(),
                    "sentence": " ".join(text[start:end + 1 if end > 0 else m.end()].split())})
    return out


def change_claims(text: str) -> list[dict]:
    """Sentences that describe a result changing after the event, with the dates they mention."""
    out = []
    for m in CHANGE_RE.finditer(text):
        s = " ".join(m.group(0).split())
        kind = re.search(CHANGE_KEY, s, re.I).group(0).lower()
        kind = ("original_result" if kind.startswith(("originally", "initially")) else
                "medal_reallocated" if kind.startswith(("reallocated", "re-allocated", "upgraded", "later awarded")) else
                "doping_retest" if kind.startswith(("retest", "re-test", "reanaly")) else
                "disqualification")
        if re.search(r"\b(no one|nobody) was disqualified\b", s, re.I):
            continue
        out.append({"kind": kind, "sentence": s, "dates": DATE_RE.findall(s)})
    return out


def event_versions(event: dict, text: str) -> dict:
    """All versions of an event's key facts with the code's resolution."""
    record = {"competitors": event.get("competitors", -1), "nations": event.get("nations", -1),
              "gold": event.get("gold", ""), "silver": event.get("silver", ""), "bronze": event.get("bronze", "")}
    counts = count_claims(text)
    conflicts = []
    for c in counts:
        for attr in ("competitors", "nations"):
            if record[attr] > 0 and c[attr] != record[attr]:
                conflicts.append({"attribute": attr, "official": record[attr], "text_value": c[attr],
                                  "sentence": c["sentence"],
                                  "resolution": f"official record {record[attr]} used; the article text says "
                                                f"{c[attr]} ({c['sentence']})"})
    changes = change_claims(text)
    return {"record": record, "counts": counts, "count_conflicts": conflicts, "changes": changes}
