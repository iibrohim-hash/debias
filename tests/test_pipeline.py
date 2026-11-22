from src import pipeline


def test_sentence_start_word_is_not_a_new_name():
    # Real case from the first full run: the rewrite added "The" at the start.
    original = '"Friends" actor David Schwimmer called out some advocates.'
    revised = 'The "Friends" actor David Schwimmer called out some advocates.'
    assert pipeline.fact_check(original, revised) == {"ok": True, "new_numbers": [], "new_names": []}


def test_new_numbers_and_names_still_fail():
    out = pipeline.fact_check("Troops fired near the border.", "Israeli troops fired 20 shots near the border.")
    assert not out["ok"] and out["new_numbers"] == ["20"] and out["new_names"] == ["Israeli"]


def test_pipeline_runs_end_to_end_in_fake_mode():
    result = pipeline.run("Hamas terrorists attacked the town.", log=False)
    assert result["decision"] in ("approve", "escalate", "retry", "block", "unchanged")
