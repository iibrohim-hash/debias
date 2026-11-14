# Prompts

Owner: Madina (Person B). One file per prompt, plain text, loaded by src/rewriter.py and src/reviewers.py.

- `rewriter_system.txt`: wording only, never facts; evidence-only rule (R01); JSON output.
- `reviewer_template.txt`: one template with `{perspective}` as a parameter (palestinian, israeli, ukrainian, russian). Cite exact phrases, check against the rubric, no nationality role-play.
- `evaluator_system.txt`

Keep prompts symmetric between sides. Commit every change so results can be traced to a prompt version.
