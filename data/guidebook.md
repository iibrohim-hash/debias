# Neutral wording guidebook

Owner: Amen (Person C). Retrieved by src/rag.py, which splits this file at "## " headings,
so each entry must make sense on its own. Rebuild the RAG index after every change.

Format for every entry:

    ## R0X Short title
    Rule: one or two sentences.
    Biased: example using "Side A" / "Side B".
    Neutral: corrected example.

Every rule must work for bias toward either side (symmetry review).

## R01 Evidence-only attribution
Rule: Name who caused a harm only if the source text states it. Otherwise keep the uncertainty or attribute the claim to its source ("according to ...").
Biased: Side A forces shelled the market, killing 12 people.
Neutral: A strike hit the market, killing 12 people, according to Side B officials.
