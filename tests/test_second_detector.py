from src import config, second_detector


def test_fake_scores_drop_when_loaded_words_go():
    before, after = second_detector.scores(["Hamas terrorists attacked.", "Hamas members attacked."])
    assert before > after
    assert second_detector.status(before) == "biased"
    assert second_detector.status(after) == "unbiased"


def test_default_run_name_matches_training():
    assert second_detector.DEFAULT_RUN == "mdeberta-v3-base_seed42"
    assert second_detector.thresholds() == dict(config.THRESHOLDS)
