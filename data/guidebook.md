# Neutral wording guidebook

Owner: Amen (Person C). Retrieved by src/rag.py, which splits this file at "## " headings,
so each entry must make sense on its own. Rebuild the RAG index after every change.

Format for every entry:

    ## R0X Short title
    Rule: one or two sentences.
    Biased: example using "Side A" / "Side B".
    Neutral: corrected example.

Every rule must work for bias toward either side (symmetry review).
The rewriter changes wording only: never facts, numbers, dates, places, people or events.

Where the rules come from (this table is above the first entry, so RAG never retrieves it).
FigNews was used to find which bias types occur and to test retrieval; the rules themselves rest on the
sources below. "Team rule" means no external source has been checked yet. Before citing a source in the
report, check the wording against its current online version.

| Rule | Source |
|---|---|
| R01 | Team rule (core design decision); in line with Reuters Handbook: contentious information must be sourced |
| R02 | BBC Editorial Guidelines: do not use "terrorist" without attribution; Wikipedia Words to watch: contentious labels |
| R03 | Reuters Handbook: avoid hype words that overstate ("flood", "surge") |
| R04 | ICCPR Art. 20(2); UN Rabat Plan of Action (2012) on incitement to hatred |
| R05 | Reuters Handbook: "claimed" can suggest doubt; Wikipedia Words to watch: prefer "said" over "claimed", "admitted" |
| R06 | Reuters Handbook: "rebut", "refute" imply editorial judgment; Wikipedia Words to watch: "revealed", "pointed out" |
| R07 | Wikipedia Words to watch: "so-called", "supposed", "alleged", scare quotes |
| R08 | Team rule |
| R09 | Team rule |
| R10 | Wikipedia Words to watch: what people say, think or believe must be clearly attributed |
| R11 | Team rule |
| R12 | Reuters Handbook: death tolls in conflict must be sourced |
| R13 | Team rule |
| R14 | Hague Regulations (1907) Art. 42 on occupation; ICJ advisory opinion of 19 July 2024 |
| R15 | Team rule |
| R16 | Team rule |
| R17 | Wikipedia Words to watch: editorializing words ("notably", "clearly") |
| R18 | Team rule |
| R19 | Team rule |
| R20 | Genocide Convention (1948) Art. II; Rome Statute Arts. 6 to 8: these are legal findings, so they are attributed |
| R21 | Team rule |
| R22 | BBC Editorial Guidelines: do not adopt other people's language as our own |
| R23 | Wikipedia Words to watch: euphemisms such as "collateral damage" |
| R24 | BBC Editorial Guidelines: do not adopt other people's language as our own (applied to "martyr", "hero") |
| R25 | BBC Editorial Guidelines: do not adopt other people's language as our own (applied to place names) |

Decision on legal designations: some governments legally designate groups as terrorist organizations, while
editorial standards such as the BBC's avoid the label in the narrator's voice. We follow the editorial
standard, and state a designation only as an attributed fact (R02).

## R01 Evidence-only attribution
Rule: Name who caused a harm only if the source text states it. Never add, guess or strengthen an actor or a motive; if the text does not say who did it, keep the uncertainty or keep the attribution the text already gives ("according to ...").
Biased: Side A forces deliberately shelled the market, killing 12 people. (The source says a strike hit the market and Side B officials blamed Side A forces.)
Neutral: A strike hit the market, killing 12 people; Side B officials blamed Side A forces.

## R02 Loaded labels for groups and people
Rule: Do not use labels such as terrorist, terrorists, terror group, thugs, regime, jihadists or occupiers in the narrator's own voice. Use the group's name or a plain description such as members, fighters, forces or government. Labels used by a named speaker stay attributed to that speaker. A legal designation may be kept only as an attributed fact, for example "which the Side A government designates as a terrorist organization", and only if the text states it.
Biased: Side B terrorists attacked the checkpoint, and the Side A regime responded.
Neutral: Side B fighters attacked the checkpoint, and the Side A government responded.

## R03 Emotional intensifiers
Rule: Remove adjectives and adverbs that tell the reader how to feel, such as brutal, horrific, relentless, brazen, grim, stark, shocking, alarming or devastating, when they are the narrator's own words. Keep the facts they describe.
Biased: After weeks of relentless bombing, residents faced a grim reality, while Side B's brutal attack shocked the region.
Neutral: After weeks of bombing, residents faced difficult conditions, while Side B's attack shocked the region.

## R04 Dehumanizing language
Rule: Never describe people or groups as animals, savages, barbarians, vermin, monsters or evil in the narrator's voice, and remove words such as barbaric or savage that cast a group as less than human. Keep such words only inside a direct quote attributed to its speaker.
Biased: The hostages are held by Side A savages after their barbaric raid.
Neutral: The hostages are held by Side A after its raid.

## R05 Neutral verbs of saying
Rule: Use the same neutral verb, such as says or said, for statements from both sides. Do not use claims, alleges, insists or admits for one side only, because these signal doubt or guilt.
Biased: Side A says the building was a weapons depot, while Side B claims it was a school.
Neutral: Side A says the building was a weapons depot, while Side B says it was a school.

## R06 Verbs that presuppose truth
Rule: Do not use verbs that decide who is right, such as refuted, debunked, exposed, revealed, discredited or proved, unless the text shows the matter was settled. Use neutral verbs such as disputed, challenged, described or said.
Biased: A Side B researcher refuted Side A's account of the strike and exposed the truth.
Neutral: A Side B researcher disputed Side A's account of the strike and gave a different account.

## R07 Skeptic words and scare quotes
Rule: Do not add doubt in the narrator's voice with words such as supposed, so-called, self-proclaimed or alleged-only-for-one-side, or with quotation marks around one side's ordinary terms. Report what was said or found plainly.
Biased: Side A soldiers searched homes in the so-called safe zone for supposed weapons.
Neutral: Side A soldiers searched homes in the safe zone for weapons.

## R08 Hidden agency and passive voice
Rule: If the text states who did something, do not hide the actor behind the passive voice or verbs such as died or lost their lives. Keep the actor the text names; never remove it.
Biased: Two Side B civilians died near Side A troops. (The source says Side A troops killed them.)
Neutral: Side A troops killed two Side B civilians.

## R09 Same verbs for comparable harm
Rule: Describe comparable harm to both sides with the same verbs and nouns. Do not write murdered, slaughtered or executed for one side's dead and killed or died for the other's.
Biased: Side A civilians were murdered, while Side B civilians were killed.
Neutral: Side A civilians were killed, and Side B civilians were killed.

## R10 Unattributed opinions in the narrator's voice
Rule: Opinions, predictions and judgments such as "a necessity", "a failure" or "the real reality" must be attributed to the person or source the text names. If the text names no source, remove the judgment instead of inventing one.
Biased: An analyst praised Side A's plan. Taking the south is an absolute military necessity.
Neutral: An analyst praised Side A's plan, saying that taking the south is a military necessity.

## R11 Causation and motive framing
Rule: Do not describe one side's actions as retaliation, a response or self-defense, or the other side's as unprovoked or aggression, unless a named source says so. State the sequence of events without assigning justification.
Biased: Side A's retaliatory strikes followed Side B's unprovoked aggression.
Neutral: Side A's strikes followed Side B's attack.

## R12 Casualty figures and their sources
Rule: Keep every number exactly, together with the source the text gives for it. Do not drop the source, and do not add qualifiers such as "run by" or "unverified" to one side's sources only.
Biased: The strike killed 40 people. (The source says: 40 people were killed, according to the Side B health ministry.)
Neutral: The strike killed 40 people, according to the Side B health ministry.

## R13 Direct quotes stay unchanged
Rule: Never edit words inside quotation marks or reported speech of a named speaker, even if they are loaded. Make sure the quote stays attributed to its speaker so the reader knows it is not the narrator's view.
Biased: The Side A minister called Side B "fighters". (A wrong rewrite: the minister actually said "animals".)
Neutral: The Side A minister called Side B "animals".

## R14 No false balance
Rule: Do not add "both sides" hedging, invented symmetry or extra context. Keep terms that reflect a clear legal or factual consensus, such as occupied territory under international law or documented events; only replace the narrator's own loaded words, such as stolen land.
Biased: Settlers live on stolen Side B land, which some call occupied. (Added hedging and a loaded word.)
Neutral: Settlers live on occupied Side B land.

## R15 Same identifiers for comparable people
Rule: Describe comparable people on both sides with the same terms. Do not call one side's under-18s children and the other's teenagers or minors, or one side's dead civilians and the other's people.
Biased: Three Side A children and three Side B teenagers were among the dead.
Neutral: Three Side A children and three Side B children were among the dead.

## R16 Collective blame
Rule: Do not blame a whole population or nationality for the actions of an armed group or a government, unless the text states it. Name the group or institution that acted.
Biased: Side B launched rockets at the city. (The text says an armed group in Side B fired them.)
Neutral: An armed group in Side B launched rockets at the city.

## R17 Editorial adverbs and commentary
Rule: Remove sentence-level commentary that tells the reader how to judge events, such as disturbingly, shockingly, tellingly, notably or predictably, and evaluative words such as derogatory or shameful in the narrator's voice. Keep the description of what happened.
Biased: Disturbingly, a Side A soldier then went through the belongings, making derogatory comments.
Neutral: A Side A soldier then went through the belongings, making comments.

## R18 Terminology: massacre, slaughter, mass murder
Rule: Do not use massacre, slaughter, mass murder or carnage in the narrator's voice for either side. Use attack or killing, or attribute the term to the source that uses it. Keep every fact about who was killed.
Biased: The Side A massacre of innocent Side B villagers sparked protests.
Neutral: The Side A attack on Side B villagers sparked protests.

## R19 Terminology: clashes
Rule: Do not use clashes or fighting when the text describes one side attacking or firing on the other; describe what the text says each side did. Use clashes only when the text reports an exchange between two armed sides.
Biased: Twelve Side B protesters died in clashes with Side A troops. (The text says troops fired on protesters.)
Neutral: Twelve Side B protesters were killed when Side A troops fired on them.

## R20 Terminology: genocide, war crimes, terrorism, ethnic cleansing
Rule: Legal and contested labels such as genocide, war crimes, ethnic cleansing, terror attack or crimes against humanity must be attributed to whoever uses them, such as a court, government, group or expert. Report court rulings as what the court said or ordered, not as a verdict the narrator draws.
Biased: Side B's war crimes against Side A civilians, documented by a UN panel, continued this week.
Neutral: Side B's actions against Side A civilians, which a UN panel described as war crimes, continued this week.

## R21 Terminology: framing of the war
Rule: Do not frame a war as one side's war on a whole place or people, such as "the war on Side B" or "the assault on Side A". Name the actor and the place: "Side A's military campaign in Side B" or "Side B's attack on Side A". Keep the actor the text states.
Biased: Side A's war on Side B has destroyed thousands of homes.
Neutral: Side A's military campaign in Side B has destroyed thousands of homes.

## R22 Terminology: official operation names and official terms
Rule: Names and terms that one side's government or armed group created, such as operation names, "special operation", "flood" or "liberation", must be attributed to that side or put in quotation marks with their source. Do not adopt them as the narrator's own description.
Biased: Side A's defence ministry announced the liberation of the town, where its forces took 200 prisoners.
Neutral: Side A's defence ministry announced what it called the "liberation" of the town, where its forces took 200 prisoners.

## R23 Terminology: euphemisms that hide harm
Rule: Replace euphemisms that hide death or destruction, such as neutralized, eliminated, collateral damage, cleared or lost their lives, with plain words when the text states what happened. Keep the actor the text names.
Biased: Side B fighters were neutralized and several buildings were cleared.
Neutral: Side B fighters were killed and several buildings were destroyed.

## R24 Terminology: martyrs and heroes
Rule: Do not use martyr, martyred, hero or fallen hero in the narrator's voice for either side's dead. Use killed or died as the text supports, or attribute the term to the source that uses it.
Biased: Five Side B martyrs were killed in the strike.
Neutral: Five Side B people were killed in the strike.

## R25 Terminology: contested place and territory names
Rule: Use widely accepted geographic names in the narrator's voice. Names preferred by one side for a contested area must be attributed to that side. Keep consensus legal status terms such as occupied when the text uses them.
Biased: Side A officials toured the liberated territories, the ministry said.
Neutral: Side A officials toured what the ministry called the "liberated" territories.
