"""Parser regression tests: facts come from the infobox exactly, nothing is invented or mis-assigned."""
from agr.parse import parse_event, olympic_infobox, tournament_fields, split_names, norm

TENNIS = """[Infobox tennis tournament event]
  champ: CHI Nicolás Massú
  runner: USA Mardy Fish

[Infobox Olympic event]
  event: Men's singles tennis
  games: 2004 Summer
  venue: Athens Olympic Tennis Centre, Athens
  dates: 15–22 August 2004
  competitors: 64
  nations: 32
  gold: Nicolás Massú
  silver: Mardy Fish
  bronze: Fernando González
  prev: 2000
  next: 2008

The men's singles ..."""


def test_second_olympic_infobox_is_used():  # tennis articles were missing from the graph before this fix
    f = olympic_infobox(TENNIS)
    e = parse_event({"doc_id": "Q1", "title": "Tennis at the 2004 Summer Olympics – Men's singles"}, f)
    assert (e.venue, e.date, e.gold, e.silver, e.nations) == \
        ("Athens Olympic Tennis Centre, Athens", "15–22 August 2004", "Nicolás Massú", "Mardy Fish", 32)


def test_football_venue_count_is_not_a_venue():
    f = tournament_fields({"venues": "7", "dates": "3–19 August", "num_teams": "12", "champion_other": "GER",
                           "second_other": "SWE", "third_other": "CAN"})
    assert f["venue"] == "" and f["gold"] == "GER" and f["nations"] == "12"


def test_team_names_split_without_breaking_mc_names():
    assert split_names("Dani KingLaura TrottJoanna Rowsell") == ["Dani King", "Laura Trott", "Joanna Rowsell"]
    assert split_names("David McKeon") == ["David McKeon"]


def test_norm_dash_and_accent_insensitive():
    assert norm("Riocentro – Pavilion 4") == norm("Riocentro - Pavilion 4")
    assert norm("Kökény") == "kokeny"


from agr.parse import person_names


def test_relay_lists_with_marks_and_footnotes():
    assert split_names("Caeleb Dressel, Michael Phelps, Ryan Held, Nathan Adrian, Jimmy Feigen*, Anthony Ervin*") == \
        ["Caeleb Dressel", "Michael Phelps", "Ryan Held", "Nathan Adrian", "Jimmy Feigen", "Anthony Ervin"]
    assert "Ian Crocker" in split_names("USAIan Crocker, Michael Phelps, Neil Walker *Indicates the swimmer only competed")
    assert split_names("BLR Victoria Azarenka, Max Mirnyi") == ["Victoria Azarenka", "Max Mirnyi"]


def test_unsplittable_teams_create_no_athletes():
    assert person_names("Viktor Ahn Semion Elistratov Vladimir Grigorev Ruslan Zakharov") == []
    assert person_names("CAN (4th title)") == []
    assert person_names("USA") == []
    assert person_names("Dani KingLaura TrottJoanna Rowsell") == ["Dani King", "Laura Trott", "Joanna Rowsell"]
    assert person_names("Michael Phelps") == ["Michael Phelps"]
    assert person_names("Juan Martín del Potro") == ["Juan Martín del Potro"]


def test_event_phrase_match_does_not_confuse_men_and_women():
    from agr.tools import event_phrase_match
    assert event_phrase_match("Men's 20 kilometres walk", norm("Athletics at the 2012 Summer Olympics – Men's 20 kilometres walk"))
    assert not event_phrase_match("Men's 20 kilometres walk", norm("Athletics at the 2012 Summer Olympics – Women's 20 kilometres walk"))


def test_read_document_resolves_a_title_exactly():
    from agr.tools import Toolbox
    tb = Toolbox.__new__(Toolbox)
    tb.docs = {"Q5198385": {"title": "Cycling at the 2000 Summer Olympics – Men's individual pursuit"}}
    assert tb._resolve_doc("Cycling at the 2000 Summer Olympics – Men's individual pursuit") == "Q5198385"
    assert tb._resolve_doc("Q5198385") == "Q5198385"
    assert tb._resolve_doc("Cycling at the 2000 Summer Olympics") == "Cycling at the 2000 Summer Olympics"  # no fuzzy match
