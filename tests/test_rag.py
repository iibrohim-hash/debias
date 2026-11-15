"""retrieve() on a small temporary guidebook. Runs with only pytest installed (DEBIAS_FAKE=1)."""
import pytest

from src import config, contract, rag

GUIDEBOOK = """# Test guidebook

## R01 Evidence-only attribution
Rule: Name who caused a harm only if the source text states it.
Biased: Side A forces shelled the market.
Neutral: A strike hit the market, according to Side B officials.

## R02 Loaded labels
Rule: Replace labels such as terrorists or regime with neutral descriptions.
Biased: Side B terrorists attacked the post.
Neutral: Side B fighters attacked the post.

## R03 Emotive verbs
Rule: Replace emotive verbs such as butchered or slaughtered with plain verbs.
Biased: Side A soldiers butchered villagers.
Neutral: Side A soldiers killed villagers.

## R04 Casualty framing
Rule: Report deaths with the same wording for both sides.
Biased: Side A civilians were murdered while Side B people died.
Neutral: Side A and Side B civilians were killed.

## R05 Hidden agency
Rule: Do not hide an actor stated in the source behind the passive voice.
Biased: Mistakes were made at the crossing.
Neutral: Side B police made mistakes at the crossing, the report said.
"""


@pytest.fixture
def guidebook(tmp_path, monkeypatch):
    path = tmp_path / "guidebook.md"
    path.write_text(GUIDEBOOK, encoding="utf-8")
    monkeypatch.setattr(config, "GUIDEBOOK_PATH", path)
    return path


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 10])
def test_contract_and_no_duplicates(guidebook, k):
    out = contract.validate_retrieve(rag.retrieve("terrorists butchered civilians", k), k)
    assert rag.entry_id(out[0]) == config.ALWAYS_INCLUDED_RULE
    ids = [rag.entry_id(e) for e in out]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("query", ["", "zzz unrelated words"])
def test_r01_without_any_match(guidebook, query):
    out = contract.validate_retrieve(rag.retrieve(query, 3), 3)
    assert rag.entry_id(out[0]) == config.ALWAYS_INCLUDED_RULE


def test_r01_in_top_k_is_not_added_twice(guidebook):
    assert rag.retrieve_ids("source text states who caused a harm", 3)[0] == "R01"
    assert len(rag.retrieve("source text states who caused a harm", 3)) == 3


def test_obvious_match_ranks_first(guidebook):
    assert rag.retrieve_ids("butchered", 3)[1] == "R03"
    assert rag.retrieve_ids("terrorists regime", 3)[1] == "R02"


def test_entries_keep_full_text(guidebook):
    out = rag.retrieve("passive voice", 1)
    assert out[1].startswith("## R05 Hidden agency") and "Neutral:" in out[1]


def test_rank_covers_every_entry(guidebook):
    assert sorted(rag.entry_id(e) for e in rag.rank("anything")) == ["R01", "R02", "R03", "R04", "R05"]


def test_guidebook_change_is_picked_up(guidebook):
    assert "R06" not in rag.retrieve_ids("ceasefire", 3)
    guidebook.write_text(GUIDEBOOK + "\n## R06 Ceasefire wording\nRule: Say who broke a ceasefire only if stated.\n")
    assert rag.retrieve_ids("ceasefire", 3)[1] == "R06"
