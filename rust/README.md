# hr-rag: Rust RAG chatbot and evaluator

A self-contained Rust implementation of the retrieval-augmented HR assistant for the Suryodaya corpus, with an
evaluation harness that runs the 1,000-question multi-hop test set and reports accuracy per language and per hop type.
Generation and embeddings run on a local [Ollama](https://ollama.com) server, so everything stays on your machine.

This is phase 1 (RAG baseline). Phase 2 fine-tunes a small model on training data built from the corpus and compares it
against this baseline with the same harness.

## What is in it

| module | role |
|---|---|
| `text.rs` | Unicode normalisation (Indic digits to ASCII), Indic-aware tokenizer (keeps matras and virama inside words), char 4-grams, sentence splitter, script-based language detection |
| `chunk.rs` | section-aware chunker (heading path kept on every chunk, max 220 words); identical output to `rag/ragcore.py` |
| `index.rs` | BM25 over word tokens and char 4-grams (k1 1.4, b 0.75), optional dense vectors; saved as one bincode file |
| `retrieve.rs` | `bm25`, `dense` or `hybrid` retrieval, fused with reciprocal rank fusion (k 60), with language / document filters |
| `ollama.rs` | minimal HTTP client for `/api/embed` and `/api/chat` (JSON-schema constrained output, temperature 0, fixed seed) |
| `session.rs` | owned API used by the notebook: `Session` (open, ensure_models, embed, search, ask, retrieval, eval, score), HTML reports and run comparison |
| `data.rs` | native file window to choose the data; finds the corpus and test set in a zip (extracting it), a folder, or from any file inside it; re-opens on a wrong pick |
| `rag.rs` | `simple` strategy (retrieve with the question, answer once) and `agentic` strategy (plan per-fact queries in each document's language, retrieve per query, answer, re-plan for missing facts) |
| `eval.rs` | gold-evidence mapping, numeric-match accuracy, evidence recall, citation recall, resumable prediction files |

Checked against the Python reference in this repository: the 5,507 chunks are byte-identical, the gold evidence ids for
all 1,000 questions are identical to `results/gold_chunks.json`, and single-shot BM25 retrieval reproduces the published
baseline (fact recall@10 80.9%, all facts@10 64.0%; cross-language all facts@10 4.0%).

## Jupyter

`../notebooks/hr_rag_chatbot.ipynb` is self-contained: it carries this library's code in its own cells (generated from
`src/` by `tools/notebook/make_notebook.py`), so it can also be copied anywhere on its own. Open it with the **Rust**
(evcxr) kernel, start the Ollama app and *Run All*. It opens the standard Windows file window (on top, starting in
Downloads) to choose the data: the repository zip, or any file inside the repository folder. It extracts the zip if
needed and writes its index and results to a `hr_rag_work` folder beside the documents.

The library API it uses, also usable from any Rust code:
```rust
use hr_rag::*;
let d = hr_rag::data::locate("C:/Users/me/Downloads/suryodaya-indic-hr-mhqa.zip")?;
let mut s = Session::open_dirs(&d.corpus, d.testset.as_deref(), &d.work, "http://localhost:11434")?;
s.ensure_models(&["bge-m3", "gemma3:4b"])?;
s.embed("bge-m3")?;
let a = s.ask("नोटिस अवधि कितनी है?", &Config::new("gemma3:4b"))?;
let r = s.eval(&Config::new("gemma3:4b"), 10)?;   // every 10th question
```

## Command line (alternative)

## Setup (Windows, macOS or Linux)

1. Install Rust (`rustup`) and Ollama. On Windows either the MSVC or the GNU toolchain works; the crate has no C dependencies.
2. Pull models:
   ```
   ollama pull bge-m3        # multilingual embeddings, covers all 8 languages
   ollama pull gemma3:4b     # generator (about 3.3 GB, fits an 8 GB GPU with room for a 12k context)
   ollama pull gemma3:1b     # optional smaller generator for comparison
   ```
3. Build and index (run from this `rust` folder):
   ```
   cargo build --release
   target/release/hr-rag index
   target/release/hr-rag embed --model bge-m3
   ```
   On Windows the binary is `target\release\hr-rag.exe`.

## Use

Chat:
```
target/release/hr-rag chat --model gemma3:4b --verbose
```
Ask in any of the 8 languages. The answer comes in the language of the question, with the document sections it used.

Search only:
```
target/release/hr-rag search "মাতৃত্বকালীন ছুটি" --mode hybrid --k 5
target/release/hr-rag search "notice period" --lang Tamil --mode dense
```

## Measuring accuracy

Quick run on 100 questions (every 10th, which keeps the language and hop-type mix), then the full set. Runs append to
the output file and resume where they stopped, so they can be interrupted safely.
```
target/release/hr-rag eval --model gemma3:4b --strategy agentic --mode hybrid --every 10 --out runs/gemma3-4b_agentic_hybrid.jsonl
target/release/hr-rag eval --model gemma3:4b --strategy agentic --mode hybrid --out runs/gemma3-4b_agentic_hybrid.jsonl
```
Suggested baseline grid (all on the same 100 questions first):

| run | flags |
|---|---|
| simple BM25 | `--strategy simple --mode bm25` |
| simple hybrid | `--strategy simple --mode hybrid` |
| agentic hybrid | `--strategy agentic --mode hybrid` |
| agentic hybrid, 1B | `--model gemma3:1b --strategy agentic --mode hybrid` |

`scripts/run_baseline.ps1` (Windows) and `scripts/run_baseline.sh` run this grid and write one report per run.

Score a finished file again, write a markdown report, or export items for an LLM / human judge:
```
target/release/hr-rag score runs/gemma3-4b_agentic_hybrid.jsonl --report runs/gemma3-4b_agentic_hybrid.md --export-judge runs/judge_in.json
target/release/hr-rag score runs/gemma3-4b_agentic_hybrid.jsonl --judge runs/judge_out.json
```
The judge file format is `{"BN-001": {"label": "correct"}, ...}` with labels `correct`, `partial` or `incorrect`
(the same format as `results/judge_verdicts.json`).

Retrieval only (no generator needed):
```
target/release/hr-rag retrieval --mode bm25 --k 10
target/release/hr-rag retrieval --mode hybrid --k 10
```

### Metrics in the report

- **numeric match**: every number in the reference short answer also appears in the system's answer. Automatic and
  strict on figures, but lenient on wording; it is computed only for questions whose short answer contains a number (about 95% of them). Use the
  judge column for the headline figure in a paper.
- **judge correct**: share labelled `correct` by an external judge (only when `--judge` is given).
- **all facts retrieved / fact recall (retrieved)**: whether the chunks put in front of the generator contained every /
  each gold supporting fact.
- **all facts cited**: whether the generator's citations cover every gold fact.
- **LLM calls/q, s/q, errors**: cost and failures (an error is a failed Ollama call; it counts as a wrong answer).

Groups: one row per question language, a `Cross-language` row for the 100 cross-language questions, and one row per
hop type.

## Expected time on an 8 GB laptop GPU

Rough figures for gemma3:4b: the agentic strategy makes 2 to 4 model calls per question with up to 12 passages in
context, so expect on the order of 10 to 30 s per question, i.e. 20 to 50 minutes for the 100-question run and several
hours for all 1,000. The simple strategy makes one call per question.

## Options

`--k` chunks per query (6), `--max-ctx` chunks in the answer prompt (12), `--rounds` agentic re-planning rounds (2),
`--num-ctx` model context in tokens (12288; lowering it below about 8k cuts passages off), `--store` index file,
`--repo` repository root (default `..`), `--ollama` server URL.
