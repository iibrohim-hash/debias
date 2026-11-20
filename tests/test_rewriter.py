"""CPU-only tests for the rewriter's parsing, validation and retry logic.

No model is downloaded: rewrite_detailed() gets hand-written outputs through
`generate_fn`.
"""
import json

import pytest

from src import config, rewriter
from src.rewriter import find_new_facts, parse_output, rewrite_detailed, validate_output

TEXT = "Side A terrorists butchered 12 civilians in the market, officials said."
SPANS = ["terrorists", "butchered"]
RULES = ["## R01 Evidence-only attribution\nRule: Name who caused a harm only if the text states it."]
GOOD = {
    "revised": "Side A fighters killed 12 civilians in the market, officials said.",
    "changes": [
        {"original": "terrorists", "revised": "fighters", "reason": "R02: contested label"},
        {"original": "butchered", "revised": "killed", "reason": "R03: loaded verb"},
    ],
}
GOOD_RAW = json.dumps(GOOD)


# --- parsing ---------------------------------------------------------------
def test_parse_valid_json():
    assert parse_output(GOOD_RAW) == GOOD


def test_parse_fenced_json():
    assert parse_output(f"```json\n{GOOD_RAW}\n```") == GOOD
    assert parse_output(f"```\n{GOOD_RAW}\n```") == GOOD


def test_parse_takes_first_object_after_chatter():
    raw = f"Here is the result:\n{GOOD_RAW}\nHope this helps! {{\"x\": 1}}"
    assert parse_output(raw) == GOOD


def test_parse_braces_inside_strings():
    raw = '{"revised": "a {b} c", "changes": []}'
    assert parse_output(raw)["revised"] == "a {b} c"


@pytest.mark.parametrize("raw", [
    "no json at all",
    '{"revised": "x", "changes": [',             # cut off at the token limit
    "{'revised': 'x', 'changes': []}",           # single quotes
    '{"revised": "x" "changes": []}',            # missing comma
])
def test_parse_broken_json(raw):
    with pytest.raises(ValueError):
        parse_output(raw)


# --- validation ------------------------------------------------------------
def test_validate_good():
    assert validate_output(TEXT, GOOD) == []


def test_validate_wrong_schema():
    assert validate_output(TEXT, {"text": "x"})
    assert validate_output(TEXT, {"revised": "x", "changes": [{"original": "a"}]})


def test_validate_invented_number():
    out = {"revised": GOOD["revised"].replace("12", "15"), "changes": GOOD["changes"]}
    errors = validate_output(TEXT, out)
    assert any("numbers" in e and "15" in e for e in errors)


def test_validate_invented_name():
    out = {
        "revised": "Side A fighters killed 12 civilians in the market, Reuters said.",
        "changes": GOOD["changes"] + [{"original": "officials", "revised": "Reuters", "reason": "R01"}],
    }
    errors = validate_output(TEXT, out)
    assert any("names" in e and "Reuters" in e for e in errors)


def test_validate_original_not_in_text():
    out = {"revised": GOOD["revised"], "changes": [{"original": "savages", "revised": "fighters", "reason": "R02"}]}
    assert any("not copied exactly" in e for e in validate_output(TEXT, out))


def test_validate_change_revised_not_in_output():
    out = {"revised": GOOD["revised"], "changes": [{"original": "terrorists", "revised": "militants", "reason": "R02"}]}
    assert any("does not appear" in e for e in validate_output(TEXT, out))


def test_validate_changed_text_needs_changes():
    out = {"revised": GOOD["revised"], "changes": []}
    assert any('"changes" is empty' in e for e in validate_output(TEXT, out))


def test_new_facts_ignores_sentence_start_and_possessive():
    original = "the army's statement said a strike hit the camp."
    revised = "According to the Army's statement, a strike hit the camp."
    assert find_new_facts(original, revised) == {"new_numbers": [], "new_names": []}


def test_new_facts_trailing_period_on_number():
    assert find_new_facts("It killed 12.", "12 people died, it said.")["new_numbers"] == []


# --- prompts -----------------------------------------------------------------
def test_feedback_section_only_with_feedback():
    without = rewriter.build_messages(TEXT, SPANS, RULES)[1]["content"]
    with_fb = rewriter.build_messages(TEXT, SPANS, RULES, feedback="Do not add names")[1]["content"]
    assert "REVIEWER FEEDBACK" not in without and "<<" not in without
    assert "REVIEWER FEEDBACK" in with_fb and "Do not add names" in with_fb and "<<" not in with_fb
    assert TEXT in without and '"terrorists"' in without and "R01" in without


@pytest.mark.parametrize("name", rewriter.PROMPT_FILES)
def test_prompts_are_conflict_neutral(name):
    text = (config.PROMPTS_DIR / name).read_text().lower()
    for word in ("israel", "palestin", "gaza", "hamas", "russia", "ukrain", "kremlin", "kyiv"):
        assert word not in text


# --- rewrite flow with a fake generator --------------------------------------
@pytest.fixture
def real_path(monkeypatch):
    """Run the non-fake code path; the generator is supplied by each test."""
    monkeypatch.setattr(config, "USE_FAKES", False)


def _scripted(*outputs):
    calls = []

    def gen(messages):
        calls.append(messages)
        return outputs[len(calls) - 1]
    return gen, calls


def test_empty_spans_skip_model(real_path):
    gen, calls = _scripted()
    result, info = rewrite_detailed(TEXT, [], RULES, generate_fn=gen)
    assert result == {"revised": TEXT, "changes": []} and calls == [] and info["skipped"]


def test_valid_first_try(real_path):
    gen, calls = _scripted(f"```json\n{GOOD_RAW}\n```")
    result, info = rewrite_detailed(TEXT, SPANS, RULES, generate_fn=gen)
    assert result == GOOD and info["retries"] == 0 and len(calls) == 1


def test_retry_once_then_valid(real_path):
    gen, calls = _scripted('{"revised": "broken', GOOD_RAW)
    result, info = rewrite_detailed(TEXT, SPANS, RULES, generate_fn=gen)
    assert result == GOOD and info["retries"] == 1 and len(calls) == 2
    retry_msg = calls[1][-1]["content"]
    assert calls[1][-2]["role"] == "assistant" and "rejected" in retry_msg and "JSON" in retry_msg


def test_two_failures_fall_back_and_log(real_path, tmp_path, monkeypatch):
    logged = []
    monkeypatch.setattr(rewriter.utils, "audit_log", lambda rec, path=None: logged.append(rec))
    bad = json.dumps({"revised": GOOD["revised"].replace("12", "40"), "changes": GOOD["changes"]})
    gen, calls = _scripted(bad, bad)
    result, info = rewrite_detailed(TEXT, SPANS, RULES, generate_fn=gen)
    assert result == {"revised": TEXT, "changes": []}
    assert info["fell_back"] and len(calls) == 2 and info["attempts"][0]["new_numbers"] == ["40"]
    assert len(logged) == 1 and logged[0]["event"] == "rewrite_failed"


def test_unchanged_output_returns_input_exactly(real_path):
    gen, _ = _scripted(json.dumps({"revised": TEXT + " ", "changes": []}))
    result, _ = rewrite_detailed(TEXT, SPANS, RULES, generate_fn=gen)
    assert result == {"revised": TEXT, "changes": []}


def test_extra_keys_dropped(real_path):
    out = dict(GOOD, notes="extra")
    out["changes"] = [dict(c, confidence=0.9) for c in GOOD["changes"]]
    gen, _ = _scripted(json.dumps(out))
    result, _ = rewrite_detailed(TEXT, SPANS, RULES, generate_fn=gen)
    assert result == GOOD


# --- model access: Llama only, clear reason when it cannot load -----------------
class GatedRepoError(Exception):  # same class name as huggingface_hub's
    pass


def test_no_fallback_model_in_config():
    assert not hasattr(config, "REWRITER_FALLBACK_MODEL")
    assert "llama" in config.REWRITER_MODEL.lower()


def test_access_message_no_token():
    msg = rewriter.access_error_message(config.REWRITER_MODEL, None)
    assert "HF_TOKEN" in msg and "Notebook access" in msg


def test_access_message_gated():
    msg = rewriter.access_error_message(config.REWRITER_MODEL, GatedRepoError("403 Client Error"))
    assert "licen" in msg and "gated-repos" in msg and config.REWRITER_MODEL in msg


def test_access_message_bad_token():
    msg = rewriter.access_error_message(config.REWRITER_MODEL, Exception("401 Unauthorized: Invalid user token"))
    assert "rejected" in msg


def test_check_access_without_token_raises(monkeypatch):
    pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(rewriter.utils, "get_hf_token", lambda: None)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        rewriter.check_model_access()


# --- quotes, duplicates, reference spans ----------------------------------------
def test_parse_repairs_unescaped_quotes():
    raw = ('{"revised": "The "Friends" actor spoke.", "changes": [{"original": "named "October 7", a", '
           '"revised": "x", "reason": "R01: y"}]}')
    out = parse_output(raw)
    assert out["revised"] == 'The "Friends" actor spoke.'
    assert out["changes"][0]["original"] == 'named "October 7", a'


def test_parse_keeps_valid_escaped_quotes():
    out = parse_output(json.dumps({"revised": 'a "b" c', "changes": []}))
    assert out["revised"] == 'a "b" c'


def test_parse_error_mentions_quote_escaping():
    with pytest.raises(ValueError, match="double quote"):
        parse_output('{"revised": "a", "changes": [1 2]}')


def test_duplicate_changes_listed_once(real_path):
    dup = dict(GOOD, changes=GOOD["changes"] + [GOOD["changes"][0]])
    out, _ = rewrite_detailed(TEXT, SPANS, RULES, generate_fn=lambda m: json.dumps(dup))
    assert out["changes"] == GOOD["changes"]


def test_spans_from_reference():
    biased = "Held by Hamas terrorists after their barbaric October 7 raid."
    neutral = "Held by Hamas after its October 7 raid."
    spans = rewriter.spans_from_reference(biased, neutral)
    assert spans == ["terrorists", "their barbaric"]
    assert all(s in biased for s in spans)
    assert rewriter.spans_from_reference(biased, biased) == []
