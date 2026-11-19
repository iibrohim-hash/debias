# Prompts

Owner: Madina (Person B). One file per prompt, plain text, loaded by src/rewriter.py and src/reviewers.py.

- `rewriter_system.txt`: wording only, never facts; evidence-only rule (R01); JSON output.
- `rewriter_user.txt`: slots `$text`, `$spans`, `$rules`, `$feedback`; the block between
  `<<IF_FEEDBACK>>` and `<<END_IF_FEEDBACK>>` is dropped when there is no feedback.
- `rewriter_retry.txt`: sent once when the first output fails parsing/validation (`$errors`).
- `reviewer_system.txt`: the reviewer's task, what counts as unfair, 3 worked examples (with neutral
  "Side X" / "Side Y", so no test case leaks into the prompt), JSON output. No nationality role-play.
- `reviewer_user.txt`: slots `$side`, `$other_side` (descriptions in `src/reviewers.py`), `$rubric`
  (built from `data/guidebook.md`: rule id, title and "Rule:" line of every entry), `$original`, `$revised`.
  Reviewer A and B use the same files; only the side changes.
- `reviewer_retry.txt`: sent once when the reviewer output fails parsing/validation (`$errors`).
- The evaluator is plain code (`pipeline.evaluate`), no prompt.

Reviewer prompt tuning: change one thing at a time, run `notebooks/B_reviewers.ipynb` on the dev split,
and the version (prompt hash) and metrics are logged to `results/reviewer_runs.jsonl`. Score the held-out
split once, at the end.

Keep prompts symmetric between sides. Commit every change so results can be traced to a prompt version.
