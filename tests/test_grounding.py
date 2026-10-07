"""Regression tests for the grounding check, built from real failures found in our own benchmark runs."""
from agr.grounding import check, answer_items


def ev(i, title, text, kind="graph", doc=None):
    return {"id": i, "doc_id": doc or i, "title": title, "text": text, "kind": kind}


THORPE = ev("E1", "Swimming at the 2004 Summer Olympics – Men's 400 metre freestyle",
            "date: August 14, 2004 (heats & final); gold: Ian Thorpe; silver: Grant Hackett", doc="Q1")
KLOCHKOVA = ev("E2", "Swimming at the 2004 Summer Olympics – Women's 400 metre individual medley",
               "date: August 14, 2004 (heats & final); gold: Yana Klochkova", doc="Q2")
PHELPS = ev("E3", "Swimming at the 2004 Summer Olympics – Men's 400 metre individual medley",
            "date: August 14, 2004 (heats & final); gold: Michael Phelps; silver: Erik Vendt", doc="Q3")
Q_GOLD = "Who won the gold medal in the event held at Olympic Aquatic Centre on August 14, 2004 (heats & final)?"


def test_combining_events_is_rejected():  # pub-028: three finals merged into one answer
    g = check("Ian Thorpe (Men's 400 metre freestyle), Yana Klochkova (Women's 400 IM), and Michael Phelps",
              [THORPE, KLOCHKOVA, PHELPS], Q_GOLD)
    assert not g.ok and g.problem == "combined"


def test_labelled_combination_is_rejected():  # eval-001: "Men's singles: X; Women's singles: Y"
    a = ev("E4", "Tennis at the 2004 Summer Olympics – Men's singles", "gold: Nicolás Massú", doc="Q4")
    b = ev("E5", "Tennis at the 2004 Summer Olympics – Women's singles", "gold: Justine Henin", doc="Q5")
    g = check("Men's singles: Nicolás Massú; Women's singles: Justine Henin", [a, b], "Who won the gold medal ...?")
    assert not g.ok and g.problem == "combined"


def test_single_event_answer_passes():
    g = check("Michael Phelps", [THORPE, KLOCHKOVA, PHELPS], Q_GOLD)
    assert g.ok and g.events == ["Q3"]


def test_silver_medallist_as_gold_is_rejected():
    g = check("Erik Vendt", [PHELPS], Q_GOLD)
    assert not g.ok and g.problem == "wrong_role"


def test_name_not_in_evidence_is_rejected():
    g = check("Ryan Lochte", [PHELPS], Q_GOLD)
    assert not g.ok and g.problem == "unsupported"


def test_partial_hallucination_is_rejected():  # one real name + one invented name
    g = check("Michael Phelps and Ryan Lochte", [PHELPS], Q_GOLD)
    assert not g.ok and g.problem == "unsupported"


def test_team_names_stored_run_together_pass():  # eval-043: infobox stores "Loredana DinuSimona Gherman..."
    team = ev("E6", "Fencing at the 2016 Summer Olympics – Women's team épée",
              "date: 11 August 2016; gold: Loredana DinuSimona GhermanSimona PopAna Maria Popescu")
    g = check("Loredana Dinu, Simona Gherman, Simona Pop, and Ana Maria Popescu", [team],
              "Who won the gold medal in the event held at Carioca Arena 3 on 11 August 2016?")
    assert g.ok


def test_title_prefix_is_ignored():  # pub-099: "Event title: names"
    relay = ev("E7", "Biathlon at the 2014 Winter Olympics – Men's relay",
               "gold: Erik LesserDaniel BöhmArnd PeifferSimon Schempp")
    g = check("Biathlon at the 2014 Winter Olympics – Men's relay: Erik Lesser, Daniel Böhm, Arnd Peiffer, "
              "Simon Schempp", [relay], "Who won the gold medal ...?")
    assert g.ok


def test_count_must_match_the_aggregate_field():
    agg = ev("E8", "aggregate(Biathlon, 2018 Winter)",
             "11 events in corpus; events with competitors > 73: 5; max competitors 87", kind="aggregate")
    assert check("5", [agg], "how many ...").ok
    assert not check("11", [agg], "how many ...").ok   # a number that appears, but is not the count asked for
    assert not check("87", [agg], "how many ...").ok


def test_number_in_a_date_does_not_count():
    fact = ev("E9", "Judo at the 2016 Summer Olympics – Women's 57 kg", "date: 8 August 2016; nations: 23")
    assert check("23", [fact], "How many nations competed ...?").ok
    assert not check("8", [fact], "How many nations competed ...?").ok


def test_unknown_and_uncited():
    assert check("unknown", [PHELPS]).problem == "unknown"
    assert check("Michael Phelps", []).problem == "uncited"


def test_answer_items():
    assert answer_items("A, B and C") == ["A", "B", "C"]
    assert answer_items("Men's singles: X; Women's singles: Y") == ["X", "Y"]


def test_medal_table_in_article_text():  # hard-033: the qualifying table ranked the SILVER medallist first
    chunk = ev("E10", "Cycling at the 1992 Summer Olympics – Women's individual pursuit",
               "Final classification / Results table / ! | Gold: | ! | Silver: | ! | Bronze: / Petra Rossner | "
               "Kathy Watt | Rebecca Twigg / 1 | Kathy Watt | AUS | 3:41.886 | Q | OR", kind="chunk")
    q = "Who won the gold medal in the women's individual pursuit cycling event at the 1992 Summer Olympics?"
    assert check("Petra Rossner", [chunk], q).ok
    g = check("Kathy Watt", [chunk], q)
    assert not g.ok and g.problem == "wrong_role"


def test_medal_table_with_newlines():  # the real article text puts the medallists on the next line
    chunk = ev("E11", "Cycling at the 1992 Summer Olympics – Women's individual pursuit",
               "Results table\n! | Gold: | ! | Silver: | ! | Bronze:\nPetra Rossner | Kathy Watt | Rebecca Twigg\n\n"
               "1 | Kathy Watt | AUS | 3:41.886", kind="chunk")
    q = "Who won the gold medal in the women's individual pursuit cycling event at the 1992 Summer Olympics?"
    assert check("Petra Rossner", [chunk], q).ok
    assert check("Kathy Watt", [chunk], q).problem == "wrong_role"


def test_sentence_naming_two_events_is_combined():  # pub-099 (final run): a sentence covering two tied events
    relay = ev("E12", "Biathlon at the 2014 Winter Olympics – Men's relay",
               "gold: Erik LesserDaniel BöhmArnd PeifferSimon Schempp", doc="Q12")
    xc = ev("E13", "Cross-country skiing at the 2014 Winter Olympics – Women's 30 kilometre freestyle",
            "gold: Marit Bjørgen", doc="Q13")
    a = ("For Biathlon at the 2014 Winter Olympics – Men's relay, the gold medallists were Erik Lesser, Daniel Böhm, "
         "Arnd Peiffer, and Simon Schempp. For Cross-country skiing at the 2014 Winter Olympics – Women's 30 kilometre "
         "freestyle, the gold medallist was Marit Bjørgen.")
    assert check(a, [relay, xc], "Who won the gold medal ...?").problem == "combined"


def test_quick_match_accepts_a_team_whatever_the_separators():
    from agr.benchmark import quick_match
    assert quick_match("Dani King, Laura Trott, and Joanna Rowsell", ["Dani KingLaura TrottJoanna Rowsell"])
    assert quick_match("Erik Lesser, Daniel Böhm, Arnd Peiffer, Simon Schempp (Biathlon)", ["Erik LesserDaniel BöhmArnd PeifferSimon Schempp"])
    assert not quick_match("Dani King, Laura Trott", ["Dani KingLaura TrottJoanna Rowsell"])            # one missing
    assert not quick_match("Dani King, Laura Trott, Joanna Rowsell, Ed Clancy", ["Dani KingLaura TrottJoanna Rowsell"])  # one extra


def test_parallel_quota_errors_skip_a_spent_key_only_once():
    from agr import llm
    keys, idx = llm._KEYS[:], llm._key_idx[0]
    try:
        llm._KEYS[:] = ["a", "b", "c"]; llm._key_idx[0] = 0
        assert llm._rotate_key(0) and llm._key_idx[0] == 1   # first worker moves past key a
        assert llm._rotate_key(0) and llm._key_idx[0] == 1   # second worker saw key a fail too: no further skip
        assert llm._rotate_key(1) and llm._key_idx[0] == 2
        assert not llm._rotate_key(2)                        # last key spent
    finally:
        llm._KEYS[:] = keys; llm._key_idx[0] = idx
