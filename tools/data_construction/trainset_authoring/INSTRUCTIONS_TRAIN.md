# Writing training questions for the Suryodaya HR benchmark (read fully before starting)

You are writing **training** questions for a multilingual, multi-hop RAG benchmark built on the HR-policy manuals of a
fictional company (Suryodaya Integrated Solutions Pvt. Ltd.). A separate 1,000-question **test** set already exists.
Your questions will be used to fine-tune a small model, so they must be as good as test questions, and they must use
only the facts in your packet (none of those facts appear in the test set).

Paths (all absolute):
- Train folder: `/tmp/claude-0/-home-claude/611c2011-d5a4-541d-baff-a8ff741d8359/scratchpad/train`
- Your packet: `<train folder>/packets/<P>.json` (P is given in your task)
- Write your output to: `<train folder>/out/<P>_partNN.json` (NN = 01, 02, …)
- Validator: `python3 -I <train folder>/tools/validate.py <train folder> <P>` (add `--final` at the end)
- The full documents, if you need context around a fact: `/tmp/claude-0/-home-claude/611c2011-d5a4-541d-baff-a8ff741d8359/scratchpad/repo/suryodaya-indic-hr-mhqa/corpus/<Language>/<document>.md`

## Your packet
`packet` id, `kind` (`mono` or `cross_language`), the language(s), the documents, the `quota`, and `facts`. Each fact
has a `fid`, the `document`, `doc_title`, `section`, `page`, `subject`, `aspect`, `value`, `unit`, and `evidence`, a
sentence copied verbatim from the document. **The evidence is the ground truth.** The value is a ledger paraphrase of
it. If a fact's evidence is unclear or does not support its value, do not use that fact.

## What to write
Write exactly the number of questions in `quota`, per hop type:
- `cross_document`: the two facts come from the two different documents of the packet.
- `cross_section`: the two facts come from the same document, from different top-level sections (e.g. 4.2 and 7.1).
- `within_section`: the two facts come from the same document and the same top-level section, and are distinct.
- `cross_language` (cross-language packets only): one fact from each language's document. Write exactly the number
  asked in each language given in `quota.asked_in`. The question is in that language. When you name a concept from
  the other-language document, you may add that document's own term in its own script in parentheses, as real
  bilingual staff would. Never use English.

Every question uses **exactly two facts** (`fact_ids`), and **must need both**: a reader with only one of them cannot
answer it. Make the second hop real: compare the two values, combine them in a calculation (sum, difference, ratio,
deadline arithmetic, eligibility under both rules, the stricter/looser of two limits), or apply one rule to the outcome
of the other. Good questions read like an employee, manager or HR officer asking about a concrete situation.
Hypothetical scenario figures (a salary, a joining date, a claim amount, number of dependants) are welcome when they
make the reasoning concrete.

### Absolute rules
1. **Do not reveal the tested values.** The question must not state the value of either fact, in digits or words.
   Describe the rule precisely instead ("the notice period for executive-grade staff", "the cap on relocation
   allowance per dependant") so a reader must look it up. Scenario numbers that are not fact values are allowed; if
   a scenario number happens to equal a digit in a fact's value, list it in `scenario_numbers` (the validator
   otherwise flags it) — and only do this when it is genuinely a scenario figure, never the answer.
2. **Correct and grounded.** The answer must follow from the two evidence sentences (plus your arithmetic). Check the
   arithmetic twice. Keep working days and calendar days apart. If the two facts cannot be combined into a question
   with one clear answer, choose other facts.
3. **Unambiguous.** One reading, one answer. Say whether a difference or a ratio is wanted; fix the unit; name the
   employee grade/category when a rule depends on it.
4. **Written by you, by hand.** Never generate question, answer or short_answer text with a script, template,
   string substitution or mail-merge. Scripts are allowed only to read the packet and to run the validator.
5. **Language and script.** Write in the packet language (cross-language: the `asked_in` language) in its own script.
   No English words, except acronyms the documents themselves use (HRBP, CTC, POSH …; the validator checks this).
   Use the digit style the documents of that language use: Bengali, Hindi, Marathi, Odia and Punjabi documents mostly
   use native digits; Tamil, Telugu and Kannada mostly use ASCII digits.
6. **Variety.** Use each fact at most twice in the packet, never the same pair of facts twice, and vary the question
   types and openings. Spread your questions over the whole packet, not just the first facts.

### Fields of each item
```json
{"lid": "<P>-001", "hop_type": "cross_document", "asked_in": "<Language>", "fact_ids": ["<P>.f012", "<P>.f087"],
 "question": "...", "short_answer": "...", "answer": "...", "scenario_numbers": []}
```
- `lid`: `<P>-001`, `<P>-002`, … in order, unique.
- `short_answer`: the minimal final answer, at most ~15 words: a number with unit, yes/no with the decisive figure, a
  date, a ratio or a short phrase.
- `answer`: 2–4 sentences in the same language, stating both facts with their section numbers and the reasoning, e.g.
  "निर्देशिका की धारा ४.१ के अनुसार … धारा ६.२ के अनुसार … अतः …".
- `scenario_numbers`: optional, see rule 1.

### Two examples from the test set (style only; do not reuse)
- Hindi, cross_document: "कार्यपालक-स्तर के कर्मी की परिवीक्षा-अवधि कितनी है, और कल्याण कोष के लाभों (अनुग्रह, शिक्षा एवं
  विवाह सहायता) के लिए न्यूनतम कितनी सेवा-अवधि आवश्यक है? परिवीक्षा पूरी होते ही स्थायी हुए कर्मी को इन लाभों के लिए
  स्थायीकरण के बाद कितना अतिरिक्त समय प्रतीक्षा करनी होगी?" → short answer "६ माह अतिरिक्त (परिवीक्षा ६ माह, कल्याण
  कोष हेतु १२ मास)".
- Marathi, cross_section: "एका कर्मचाऱ्याला आंतरशहरीय बदलीचा आदेश मिळाला असून तिच्यासोबत ३ अवलंबित व्यक्ती स्थलांतरित
  होणार आहेत. आदेश अंमलात येण्यापूर्वी तिला किती कामकाजाच्या दिवसांची नोटीस मिळेल, आणि तिला जास्तीत जास्त किती
  स्थलांतर भत्ता मिळू शकतो?" → "२१ कामकाजाचे दिवस नोटीस; कमाल ₹५९,००० भत्ता" (3 dependants is a scenario figure).

## How to work
1. Read your packet (it is large; read it in parts if needed). Plan which fact pairs to use before writing.
2. Write the items in files of **at most 10 items each**, one Write call per file (a JSON list).
3. After each file, run the validator (without `--final`) and fix every ERROR by editing your file.
4. When all files are written, run the validator **with `--final`**. It must print `OK`.
5. Re-read your questions once for leaked values, ambiguity and arithmetic.

Keep narration between tool calls to one line, and never paste large blocks of your own output into messages: there
is a per-message output limit, and large single outputs have crashed earlier runs.

## Report (under 150 words)
The final validator line, the number of items per hop type, any facts you skipped as unclear (fids), and anything a
reviewer should check.
