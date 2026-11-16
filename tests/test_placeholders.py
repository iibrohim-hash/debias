from src.placeholders import find_leftovers, mask_sides


def test_israel_palestine():
    text = "Israeli forces and Hamas fighters clashed in the Gaza Strip; Israel's PM Netanyahu spoke."
    out = mask_sides(text, "israel_palestine")
    assert out == "[SIDE_A] forces and [SIDE_B] fighters clashed in the [SIDE_B]; [SIDE_A]'s PM [SIDE_A] spoke."
    assert find_leftovers(out) == []


def test_russia_ukraine():
    text = "The Kremlin said Russian troops left; Zelenskyy thanked Ukrainians in Kyiv."
    out = mask_sides(text, "russia_ukraine")
    assert out == "The [SIDE_A] said [SIDE_A] troops left; [SIDE_B] thanked [SIDE_B] in [SIDE_B]."


def test_acronym_case_sensitive_and_word_boundaries():
    assert mask_sides("the pa system in Israelville", "israel_palestine") == "the pa system in Israelville"
    assert mask_sides("the PA said", "israel_palestine") == "the [SIDE_B] said"


def test_leftovers_across_conflicts():
    masked = mask_sides("Israel and Russia", "israel_palestine")
    assert find_leftovers(masked) == ["Russia"]


def test_hashtags_and_unicode():
    assert mask_sides("#IsraelHamasWar and #HamasTerrorists", "israel_palestine") == \
        "#[SIDE_A][SIDE_B]War and #[SIDE_B]Terrorists"
    assert mask_sides("#Gaza_War", "israel_palestine") == "#[SIDE_B]_War"
    assert mask_sides("#PAKISTAN", "israel_palestine") == "#PAKISTAN"
    assert mask_sides("ısrael said", "israel_palestine") == "[SIDE_A] said"  # dotless i variant


def test_leftovers_per_conflict():
    masked = mask_sides("Israel and Russia", "israel_palestine")
    assert find_leftovers(masked, "israel_palestine") == []
    assert find_leftovers(masked, "russia_ukraine") == ["Russia"]
