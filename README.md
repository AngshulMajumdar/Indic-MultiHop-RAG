# Suryodaya Indic HR Multi-hop QA

**A multilingual, multi-hop question-answering benchmark over long HR-policy documents in 8 Indian languages, with
retrieval and agentic-RAG baselines and a Rust RAG chatbot that runs on a laptop.**

An employee's question is rarely answered by one sentence. *"If I am confirmed after probation, how much longer before
I can claim the welfare fund?"* needs two clauses from two long documents and a small calculation, and in a
multilingual company the two documents may be in different languages. This repository provides data to measure that,
the code to run it, and every prediction behind the reported numbers.

Project article: `docs/index.html` (published with GitHub Pages, see [Website and Medium](#website-and-medium)).

| | |
|---|---|
| Documents | 50 HR-policy manuals of one fictional company, written natively in 8 languages |
| Size | 675,913 words; 12k–17k words per document; 3,444 page markers |
| Gold facts | 6,315, each with section, page and a verbatim evidence sentence |
| Questions | 1,000 multi-hop (710 cross-document, 180 cross-section, 100 cross-language, 10 within-section) |
| Retrieval baseline | one query finds all evidence for 64% of questions (top 10), 4% of cross-language ones |
| Agentic RAG baseline | 998 correct, 2 partial, 0 wrong (99.8% strict) |
| Chatbot | Rust, local models through Ollama, Jupyter notebook with per-language accuracy reports |
| Licences | data CC BY 4.0, code MIT |

## Contents

1. [The four parts](#the-four-parts)
2. [Repository layout](#repository-layout)
3. [Quick start](#quick-start)
4. [Data format](#data-format)
5. [Retrieval system](#retrieval-system)
6. [Results](#results)
7. [Rust chatbot and evaluator](#rust-chatbot-and-evaluator)
8. [Website and Medium](#website-and-medium)
9. [Limitations](#limitations)
10. [Citation and licence](#citation-and-licence)

## The four parts

**1. Corpus.** The HR policies of Suryodaya Integrated Solutions Pvt. Ltd., a fictional company, so no model has seen
the facts in training. The policies are spread over 50 documents: Tamil 10, Hindi 9, Telugu 7, Marathi 6, Odia 5,
Punjabi 5, Bengali 4 and Kannada 4.
- Every document was written natively in its language, not translated.
- Numbers follow local practice. Bengali, Devanagari, Gurmukhi and Odia documents mostly use native digits; Tamil,
  Telugu and Kannada mostly use Western digits.
- Topics span the employee life cycle: recruitment, onboarding, probation, compensation, leave, attendance, travel,
  remote work, performance, training, discipline, grievances, harassment prevention, whistleblowing, benefits, loans,
  retirement funds, safety, IT use and separation.
- Each document has a **fact ledger** of gold facts with verbatim evidence.
- Five **topic-collision pairs** test whether retrieval matches facts rather than topics. Each pair is two documents in
  different languages on the same topic (recruitment, travel, separation, disciplinary, training) with no shared content.

**2. Test set.** 1,000 questions, each needing at least two facts:

| Language | Docs | Questions |
|---|---|---|
| Tamil | 10 | 180 |
| Hindi | 9 | 162 |
| Telugu | 7 | 126 |
| Marathi | 6 | 108 |
| Odia | 5 | 90 |
| Punjabi | 5 | 90 |
| Bengali | 4 | 72 |
| Kannada | 4 | 72 |
| Cross-language (5 topic pairs × 20) | — | 100 |

No question states the values it tests. Each question has a short and a full reference answer, its source sections, and
supporting facts that resolve to ledger entries.

**3. Baselines.** Single-query retrieval (BM25 over word tokens and character 4-grams) and an agentic RAG system that
plans per-fact searches in each document's language. All predictions, judge verdicts and search logs are included.

**4. Rust RAG chatbot.** A Rust implementation of the whole pipeline (tokenizer, section-aware chunker, BM25 + dense
index, retrieval, simple and agentic answering, evaluation) running local models through Ollama on a laptop. It comes
with a command-line tool and a self-contained Jupyter notebook for the Rust kernel.

## Repository layout

```
corpus/<Language>/NN_<topic>.md           policy documents (page boundaries as HTML comments)
ledgers/<Language>/NN_<topic>.facts.json  fact ledger for each document
testset/suryodaya_qa_testset_1000.json    the 1,000 questions
trainset/suryodaya_qa_train.json, _dev.json  training/dev questions (facts disjoint from the test set); stats.json

rag/          Python reference retrieval (standard library only)
  ragcore.py      digit normalisation, Indic-aware tokenizer, section-aware chunking, BM25
  build_index.py  builds rag/store/ (5,507 chunks, about 1 s)
  search.py       search CLI used by the agents: language/document filters, --get with neighbours, query logs
eval/         Python scoring
  retrieval_baseline.py  gold evidence -> chunk mapping; single-query retrieval baselines
  score.py               accuracy, numeric check, cited-evidence recall, search effort
prompts/      agent protocol (agentic_rag_agent.md), grading protocol (judge.md), data-repair protocol (qa_repair.md)
results/      predictions, judge verdicts, search logs and metrics of the agentic RAG run

rust/         Rust RAG chatbot and evaluator (library + `hr-rag` CLI); see rust/README.md
notebooks/    hr_rag_chatbot.ipynb: self-contained notebook for the Rust (evcxr) Jupyter kernel
              finetune_colab.ipynb: baseline, LoRA fine-tuning and re-evaluation on Google Colab (Python)
finetune/     hrqa.py: Python retrieval, evaluation (closed-book / RAG / oracle) and LoRA training used by the Colab notebook
paper/        Neurocomputing manuscript (LaTeX); make_numbers.py fills every number and table from data/ and results/

docs/         project website for GitHub Pages: index.html, medium.html (for Medium import), assets/ (figures)
notes/        corpus design plan
tools/
  data_construction/  mechanical scripts used to assemble ledgers and the test set (no text generation)
  site/               make_figures.py and build_site.py: rebuild docs/ from results/ and data/
  notebook/           make_notebook.py: rebuild the notebook from rust/src
```

## Quick start

### Python reference (no dependencies, Python 3.9+)
```bash
python3 rag/build_index.py            # -> rag/store/ (5,507 chunks)
python3 eval/retrieval_baseline.py    # -> results/gold_chunks.json, results/baseline_retrieval.json
python3 eval/score.py                 # -> results/results_agentic_rag.json (from stored predictions and verdicts)
python3 rag/search.py "परिवीक्षा अवधि" --lang Hindi --k 3
```

### Jupyter notebook (Rust kernel)
Needs Rust, the [evcxr](https://github.com/evcxr/evcxr) Jupyter kernel and the [Ollama](https://ollama.com) app.
1. Open `notebooks/hr_rag_chatbot.ipynb` with the **Rust** kernel and start Ollama.
2. Choose *Run All*. A Windows file window opens: pick the repository zip, or any file inside the repository folder.
3. The notebook then:
   - downloads the models it needs: `bge-m3`, `gemma3:4b` and `gemma3:1b` (about 5 GB the first time);
   - builds the index and embeds the corpus;
   - shows retrieval recall;
   - answers test questions next to their reference answers;
   - opens a chat box for your own questions;
   - runs the accuracy grid on 100 stratified questions.
4. Everything it writes goes to `data/hr_rag_work/`, and interrupted runs resume where they stopped.

### Command line (Rust)
```bash
cd rust
cargo build --release
target/release/hr-rag index                      # chunk + BM25
target/release/hr-rag embed --model bge-m3       # dense vectors (Ollama)
target/release/hr-rag chat --model gemma3:4b     # chatbot, any of the 8 languages
target/release/hr-rag eval --model gemma3:4b --every 10 --out runs/gemma3-4b.jsonl
target/release/hr-rag retrieval --mode bm25 --k 10
```
The crate has no C dependencies, so it builds with the MSVC or the GNU toolchain on Windows. Details are in `rust/README.md`.

## Data format

Question record:
```json
{"id": "HI-016", "language": "Hindi", "hop_type": "cross_document",
 "question": "...", "short_answer": "...", "answer": "...",
 "source": [{"language": "Hindi", "document": "01_probation", "section": "४.१"}, ...],
 "supporting_facts": [{"language": "Hindi", "document": "01_probation", "subject": "...", "aspect": "..."}, ...]}
```
- Each `supporting_facts` key `(language, document, subject, aspect)` resolves to a ledger entry
  `{subject, aspect, value, unit, section, page, evidence}`.
- `evidence` is a verbatim substring of the document.
- Cross-language questions carry a language pair such as `Bengali-Telugu` in `language`: the languages of the two
  facts, in the order of `source`. The question is written in one of the two (the first one in 99 of 100 cases), and
  the pairs are Bengali–Telugu, Hindi–Kannada and Hindi–Telugu in both directions.

Hop types:
- `cross_document` (710): two facts from two documents.
- `cross_section` (180): two facts from two sections of one document.
- `cross_language` (100): two facts from documents in two different languages.
- `within_section` (10): two distinct facts from one section.

## Retrieval system

- **Chunking:** along section and subsection boundaries, at most 220 words per chunk. Each chunk carries its document
  title and heading path. The result is 5,507 chunks.
- **Indexes:** two BM25 indexes (k1 = 1.4, b = 0.75):
  - word unigrams after normalising all Indic digit scripts to ASCII (Indic words kept whole, vowel signs included);
  - character 4-grams, which help with morphology and compounding.
- **Fusion:** reciprocal rank fusion with k = 60.
- **Dense retrieval (Rust only):** optional bge-m3 vectors added as a third ranked list.

**Tokenizer note.** Python's `\w` does not match Indic vowel signs (matras), so `re.findall(r"\w+", text)` splits Hindi,
Tamil and other words into fragments. The tokenizer here uses `[\wऀ-ॣ०-ൿ‌‍]+`.
With the broken pattern, all-facts@10 was 0.30 overall and 0.00 for cross-language questions.

## Results

### Single-query retrieval (raw question as the query)
Recall of gold evidence chunks over the global index:

| Group | fact recall@5 | fact recall@10 | all facts@10 | all facts@20 |
|---|---|---|---|---|
| **All (1000)** | 0.714 | 0.808 | 0.638 | 0.732 |
| cross_document | 0.704 | 0.816 | 0.663 | 0.775 |
| cross_section | 0.853 | 0.921 | 0.850 | 0.917 |
| cross_language | 0.490 | 0.515 | **0.040** | **0.070** |
| cross_language, one query per language | 0.685 | 0.770 | 0.590 | 0.690 |

By question language, all facts@10 ranges from 0.556 (Punjabi) to 0.847 (Bengali). A single query almost never
retrieves both facts of a cross-language question: lexical matching finds the clause in the question's own language
only. One query per language recovers most of them.

### Agentic RAG
- **Setup:** reasoning agents (Claude instances) received batches of 40 questions and only `rag/search.py`
  (protocol: `prompts/agentic_rag_agent.md`).
- **Procedure:** agents decomposed each question, issued queries in each document's language, read neighbouring
  chunks, and cited the chunk ids they used.
- **Grading:** separate judge instances labelled each answer correct, partial or incorrect against the reference
  (`prompts/judge.md`).
- **Effort:** 3,074 tool calls, about 3 per question.
- **Search tool version:** the run used the search tool *before* the tokenizer fix. The agents compensated by
  reformulating queries.

| Group | n | Strict | Partial = ½ | Correct/Partial/Incorrect | Numeric match | Cited-fact recall | All facts cited |
|---|---|---|---|---|---|---|---|
| **All** | 1000 | **0.998** | 0.999 | 998/2/0 | 0.974 | 0.925 | 0.862 |
| Bengali | 72 | 1.000 | 1.000 | 72/0/0 | 1.000 | 0.993 | 0.986 |
| Hindi | 162 | 1.000 | 1.000 | 162/0/0 | 0.988 | 0.994 | 0.988 |
| Marathi | 108 | 0.981 | 0.991 | 106/2/0 | 0.981 | 0.870 | 0.750 |
| Odia | 90 | 1.000 | 1.000 | 90/0/0 | 0.978 | 0.814 | 0.689 |
| Punjabi | 90 | 1.000 | 1.000 | 90/0/0 | 0.932 | 0.950 | 0.900 |
| Tamil | 180 | 1.000 | 1.000 | 180/0/0 | 0.961 | 0.897 | 0.806 |
| Telugu | 126 | 1.000 | 1.000 | 126/0/0 | 0.991 | 0.933 | 0.873 |
| Kannada | 72 | 1.000 | 1.000 | 72/0/0 | 0.986 | 0.875 | 0.792 |
| Cross-language | 100 | 1.000 | 1.000 | 100/0/0 | 0.948 | 0.975 | 0.950 |

- **Numeric match:** all numbers in the reference short answer appear in the prediction (no model involved).
- **Cited-fact recall:** share of gold evidence chunks among the cited chunks. Agents sometimes cite another chunk that
  repeats the same fact, so this is a lower bound.
- **Errors:** the 2 partial answers (MR-007, MR-014) are system errors. MR-007 missed the clause that keeps maternity
  leave in the time-in-grade count. MR-014 did not retrieve the payslip query window.

### Consistency pass
The first evaluation surfaced problems in the data. All were corrected before the final numbers.
- **Documents:** contradictions inside or between documents were edited so the text says one thing:
  - Hindi 06: leave-record retention.
  - Odia 05: key vs. most-critical positions.
  - Tamil 05: IDP calendar and LDC meetings.
  - Telugu 04: rail-preference distance.
  - Telugu 05: complaint routes.
  - Telugu 07: delay-interest basis.
  - Kannada 03: contract categories.
  - Kannada 04: grade-band table.
  - Marathi 06: cross-references.
- **Questions:** every reference a judge doubted was rewritten to have a single reading (for example, stating whether a
  difference or a ratio is asked, or fixing the unit).
- **Re-run:** 27 questions were revised. These and 2 others whose evidence changed were re-answered and re-graded.
  No judge flags remain.

## Rust chatbot and evaluator

`rust/` holds a library and the `hr-rag` CLI; `notebooks/hr_rag_chatbot.ipynb` carries the same library code inline.

- **Parity:** the Rust chunker is byte-identical to the Python one (5,507 chunks), the gold-evidence mapping is
  identical for all 1,000 questions, and BM25 retrieval reproduces the recall above (all facts@10 0.640 vs 0.638;
  ties broken differently).
- **Strategies:**
  - `simple`: one search with the question, one answer.
  - `agentic`: the question's own top results plus one planned query per needed fact in the language of its
    document, and a second round for facts the model reports missing.
- **Retrieval modes:** `bm25`, `dense` (bge-m3) or `hybrid` (both fused).
- **Answers:** constrained to JSON (short answer, answer, cited chunk ids, missing information). Citations to chunks
  that were not retrieved are discarded.
- **Reports**, per language and hop type:
  - numeric-match accuracy, and judge accuracy when verdicts are supplied;
  - whether all gold facts reached the model's context, and whether the model cited them;
  - model calls and seconds per question, and errors.
- **Data selection:** the notebook opens a native file window (Windows), accepts the zip or any file inside the data,
  extracts the zip once, and rejects wrong picks with a reason.

Open-model results (gemma3 with simple and agentic strategies) are produced by the notebook on local hardware and are
not part of this release; they will be reported together with the fine-tuning experiments.

## Fine-tuning (Colab)

`notebooks/finetune_colab.ipynb` runs on Google Colab with a GPU (L4 or A100 recommended):
1. evaluates the base model (`unsloth/gemma-3-1b-it`, an ungated copy of Gemma 3 1B instruct) on all 1,000 test
   questions under three conditions: closed-book, RAG (top-6 retrieved passages) and oracle (gold passages plus
   retrieved distractors);
2. trains a LoRA adapter (rank 16, two epochs) on `trainset/` with RAFT-style contexts: the gold passages of each
   training question mixed with hard distractors from the retriever, loss on the answer only;
3. re-evaluates the fine-tuned model under the same three conditions and writes before/after tables and a figure.

Answers, the adapter and the merged model are saved to Google Drive; interrupted runs resume. Copy
`<run>_results.json` to `results/finetune/` (or type the numbers into `results/finetune/finetune_table.csv`) and run
`python3 paper/make_numbers.py && (cd paper && latexmk -xelatex main.tex)` to update the paper.

The training split is built from the 4,881 ledger facts that no test question uses, so the two splits share no facts;
`tools/data_construction/merge_train.py` asserts this when it assembles the split.

## Website and Medium

`docs/` is a static site; it needs no build step on GitHub.

- **GitHub Pages:**
  1. In the repository, open *Settings → Pages*.
  2. Set *Source: Deploy from a branch*, choose branch `main`, folder `/docs`, and save.
  3. The article appears at `https://<user>.github.io/<repo>/`. Its *Repository*, *Test set* and *Rust chatbot*
     buttons find the repository from that address automatically.
- **Medium:**
  1. After Pages is live, open Medium → *Import a story*.
  2. Paste `https://<user>.github.io/<repo>/medium.html`.
  3. That page is plain semantic HTML that Medium imports cleanly: headings, paragraphs, quotes, lists, code, and
     figures with captions. It has no tables (Medium cannot import them): the results table is an image, and the
     retrieval numbers are a list.
  4. Medium sets the canonical link to the GitHub page.
- **Rebuilding** after a change to data or results:
  ```bash
  python3 tools/site/make_figures.py    # docs/assets/*.png (needs matplotlib)
  python3 tools/site/build_site.py      # docs/index.html, docs/medium.html (numbers read from results/ and data/)
  # if an importer does not resolve relative image links:
  python3 tools/site/build_site.py --site-url https://<user>.github.io/<repo>/
  ```

## Limitations

- **One model family throughout.** The corpus, ledgers, questions and references were authored with an LLM (Claude),
  and the agentic system and judges are Claude as well. The 99.8% shows the benchmark is solvable and internally
  consistent with agentic retrieval; it is not an independent model comparison.
- **Next evaluations needed:** other model families, closed-book and oracle-evidence controls, and an independent judge.
- **No native-speaker validation yet** for any language.
- **One fictional company:** this avoids training-data contamination but limits domain diversity.
- **Uneven languages:** 72 to 180 questions per language.

## Citation and licence

- **Data** (`corpus/`, `ledgers/`, `testset/`, `trainset/`): CC BY 4.0.
- **Code** (`rag/`, `eval/`, `finetune/`, `tools/`, `rust/`, `notebooks/`, `docs/`, `paper/`): MIT.
- **Citation:** use `CITATION.cff`, or:

```bibtex
@misc{majumdar2026suryodaya,
  author = {Majumdar, Angshul},
  title  = {Suryodaya Indic HR Multi-hop QA: a multilingual multi-hop RAG benchmark over HR policies in eight Indian languages},
  year   = {2026},
  note   = {Dataset and code}
}
```
