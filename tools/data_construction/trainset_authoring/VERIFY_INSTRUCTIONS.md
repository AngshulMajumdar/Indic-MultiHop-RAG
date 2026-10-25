# Verifying training questions (read fully before starting)

You review training questions for a multilingual multi-hop HR-policy QA dataset (fictional company Suryodaya
Integrated Solutions). Another writer produced them; you did not. Your packet's work file is
`/tmp/claude-0/-home-claude/611c2011-d5a4-541d-baff-a8ff741d8359/scratchpad/train/verify_in/<P>.json`: a list of items, each with
`question`, `short_answer`, `answer`, `hop_type`, `asked_in`, and its two `facts` with the verbatim `evidence` sentence
from the source document (the evidence is the ground truth; `value` is a paraphrase of it). The full documents are in
`/tmp/claude-0/-home-claude/611c2011-d5a4-541d-baff-a8ff741d8359/scratchpad/repo/suryodaya-indic-hr-mhqa/corpus/<Language>/<document>.md` if you need the surrounding text.

For every item decide one verdict:
- `ok`: correct, needs both facts, no leaked values, one clear reading.
- `fix`: a problem you can repair by editing the question, short answer and/or answer while keeping the same two facts
  and hop type. Typical fixes: wrong arithmetic or wrong figure in the answer; the question states a tested value
  (replace it with a precise description of the rule); an ambiguity (say difference vs ratio, working vs calendar days,
  the employee grade); a short answer that does not match the answer.
- `drop`: cannot be repaired with these two facts (the evidence does not support the answer, the question does not
  really need one of the facts, or the facts contradict each other).

Rules: check every figure against the evidence and redo every calculation yourself. Keep the language and script of the
item (`asked_in`) and its digit style; no English words except acronyms the documents use. Write any replacement
text yourself, by hand — never with a script, template or string substitution. Be strict but do not rewrite items that
are fine; style preferences are not a reason to fix.

Output: write `.../train/verify_out/<P>.json` (one Write call per at most 40 items: `<P>.json`, then `<P>_b.json`,
`<P>_c.json` if needed), a JSON object keyed by lid:
  {"<lid>": {"verdict": "ok"}, "<lid>": {"verdict": "fix", "question": "<only if changed>", "short_answer": "<only if changed>",
   "answer": "<only if changed>", "reason": "<one line>"}, "<lid>": {"verdict": "drop", "reason": "<one line>"}}
Every lid in the work file must appear exactly once. Validate with a short script that the JSON parses and all lids are
covered. Keep narration to one line between tool calls; never paste large blocks of your own output into messages.

Report (under 120 words): counts of ok / fix / drop, and the most common problem.
