import pandas as pd
import pytest

from src import baseline


def test_probes_file_is_valid():
    probes = baseline.load_probes()
    assert len(probes) >= 20 and probes["id"].is_unique
    pairs = probes[probes["swap_pair_id"] != ""].groupby("swap_pair_id").size()
    assert (pairs == 2).all(), "every swap pair needs exactly two probes"
    assert set(probes["type"]) == set(baseline.PROBE_TYPES)


def test_bad_probe_type_is_rejected(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text("id,conflict,type,prompt,swap_pair_id\n1,israel_palestine,weird,Q?,\n")
    with pytest.raises(ValueError):
        baseline.load_probes(p)


def test_messages_have_the_plain_system_prompt():
    m = baseline.build_messages("What happened?")
    assert m[0] == {"role": "system", "content": baseline.SYSTEM_PROMPT}
    assert m[1] == {"role": "user", "content": "What happened?"}


def test_answer_many_batches_and_keeps_order():
    calls = []

    def fake_generate(batch):
        calls.append(len(batch))
        return [{"text": m[1]["content"].upper(), "new_tokens": 3, "hit_limit": False} for m in batch]

    out = baseline.answer_many(["a", "b", "c"], generate_fn=fake_generate, batch_size=2)
    assert [o["answer"] for o in out] == ["A", "B", "C"] and calls == [2, 1]


def test_summarize_table():
    rows = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "swap_pair_id": ["", "", "S01", "S01"],
        "decision": ["unchanged", "approve", "escalate", "approve"],
        "answer": ["a", "b", "c", "d"],
        "final": ["a", "b2", "c", "d"],
        "n_rewrites": [0, 1, 2, 1],
        "n_fact_ok": [0, 1, 1, 1],
        "second_base": [0.1, 0.9, 0.8, 0.2],
        "second_resp": [0.1, 0.2, 0.8, 0.2],
        "second_base_biased": [False, True, True, False],
        "second_resp_biased": [False, False, True, False],
    })
    t = baseline.summarize(rows)
    assert t.loc["Bias rate (second detector: share biased)"].tolist() == [0.5, 0.25]
    assert t.loc["Fact preservation (rewrites with no new numbers or names)", "Responsible"] == 0.75
    assert t.loc["Symmetry gap (swap pairs with different verdicts)"].tolist() == [1.0, 1.0]
    assert t.loc["Escalation rate (sent to a human)", "Responsible"] == 0.25
    assert t.loc["Escalation rate (sent to a human)", "Baseline"] == "n/a"
    assert t.loc["Answers changed (approved and text differs)", "Responsible"] == 0.25
    assert t.attrs["n_pairs"] == 1
