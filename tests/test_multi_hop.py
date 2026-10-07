"""Multi-hop reasoning agent: exact start, one hop at a time, no guessing on an ambiguous start."""
from agr.tools import Toolbox

EV = {
    "A16": {"v_id": "A16", "attributes": {"title": "X at the 2016 Summer Olympics – Men's 100 m", "gold": "G16"}},
    "A12": {"v_id": "A12", "attributes": {"title": "X at the 2012 Summer Olympics – Men's 100 m", "gold": "G12"}},
    "A08": {"v_id": "A08", "attributes": {"title": "X at the 2008 Summer Olympics – Men's 100 m", "gold": "G08"}},
}
PREV = {"A16": "A12", "A12": "A08"}


class FakeTG:
    def query(self, name, params):
        if name == "event_profile":
            e = EV.get(params["eid"])
            return [{"event": [e] if e else []}, {}]
        if name == "edition_hop":
            nxt = PREV.get(params["eid"]) if params["direction"] == "prev" else None
            return [{"events": [EV[nxt]] if nxt else []}]
        raise AssertionError(name)


def tb(venue_top=None):
    t = Toolbox.__new__(Toolbox)
    t.tg, t.counter, t.docs = FakeTG(), 0, {}
    t._resolve_event = lambda e: e
    if venue_top is not None:
        from agr.tools import Observation
        t.venue_events = lambda v, d, y: Observation("venue_events", "", [], bool(venue_top), venue_top)
    return t


def test_two_hops_back_reaches_the_right_edition_with_evidence_at_every_hop():
    o = tb().multi_hop(path=["prev", "prev"], event_id="A16")
    assert o.found and [c["id"] for c in o.data["chain"]] == ["A16", "A12", "A08"]
    assert len(o.evidence) == 3 and "G08" in o.evidence[-1].text


def test_missing_link_stops_the_chain_and_says_so():
    o = tb().multi_hop(path="prev,prev,prev", event_id="A16")
    assert not o.found and "no linked edition" in o.data["stopped"]


def test_ambiguous_start_takes_no_hop():
    top = [{"id": "A16", "title": "a", "date_match": 1.2}, {"id": "A12", "title": "b", "date_match": 1.2}]
    o = tb(top).multi_hop(path=["prev"], venue="Stadium", date="5 August")
    assert not o.found and len(o.data["ties"]) == 2 and o.data["chain"] == []
    assert "AMBIGUOUS" in o.summary


def test_exact_date_match_breaks_a_tie():
    top = [{"id": "A16", "title": "a", "date_match": 2.0}, {"id": "A12", "title": "b", "date_match": 1.2}]
    o = tb(top).multi_hop(path=["prev"], venue="Stadium", date="5 August")
    assert o.found and [c["id"] for c in o.data["chain"]] == ["A16", "A12"]


def test_bad_hop_is_rejected():
    assert not tb().multi_hop(path=["sideways"], event_id="A16").found


def test_date_phrase_stops_at_a_colon():
    import json
    t = Toolbox.__new__(Toolbox)
    t.catalog = lambda: {"sports": [], "venues": ["Deer Valley"], "events": {}}
    q = "The event held at Deer Valley on February 23, 2002: who won the gold medal in that same event at the previous Olympics?"
    assert t.entity_linker(q).data["date_phrase"] == "February 23, 2002"
