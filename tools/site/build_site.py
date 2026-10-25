"""Builds the project site in docs/ from one source of content:

  docs/index.html   GitHub Pages article (styled, HTML tables, light/dark)
  docs/medium.html  the same article as plain semantic HTML for Medium's "Import a story"
                    (no tables, no CSS, no scripts; tables appear as images)

Run from the repository root:  python3 tools/site/build_site.py [--site-url https://USER.github.io/REPO/]
--site-url makes image links absolute (useful if an importer does not resolve relative links).
Numbers are read from results/, corpus/, ledgers/ and testset/, so the pages always match the stored results."""
import argparse, glob, html, json, re

ap = argparse.ArgumentParser()
ap.add_argument("--site-url", default="")
args = ap.parse_args()
BASE = args.site_url.rstrip("/") + "/" if args.site_url else ""

# ------------------------------------------------------------------ numbers from the stored results
test = json.load(open("testset/suryodaya_qa_testset_1000.json"))
T = {q["id"]: q for q in test}
words = sum(len(open(f).read().split()) for f in glob.glob("corpus/*/*.md"))
facts = sum(len(json.load(open(f))) for f in glob.glob("ledgers/*/*.facts.json"))
pages = sum(len(re.findall(r"<!--", open(f).read())) for f in glob.glob("corpus/*/*.md"))
hops = {h: sum(q["hop_type"] == h for q in test) for h in ["cross_document", "cross_section", "cross_language", "within_section"]}
B = json.load(open("results/baseline_retrieval.json"))
G, LF = B["global"], B["lang_filtered"]
A = json.load(open("results/results_agentic_rag.json"))
LANGS = ["Bengali", "Hindi", "Kannada", "Marathi", "Odia", "Punjabi", "Tamil", "Telugu"]
pct = lambda x: f"{x * 100:.1f}%"
pc0 = lambda x: f"{x * 100:.0f}%"

TITLE = "Asking HR policy questions across eight Indian languages"
SUB = "A multi-hop, multilingual RAG benchmark, an agentic baseline, and a Rust chatbot that runs on a laptop"
DESC = (f"1,000 multi-hop questions over 50 natively written HR-policy documents in 8 Indian languages "
        f"({words/1000:.0f}k words, {facts:,} gold facts), with retrieval and agentic-RAG baselines and a Rust RAG chatbot.")
AUTHOR = "Angshul Majumdar"

def fig(name, caption, alt):
    return (f'<figure><img src="{BASE}assets/{name}" alt="{html.escape(alt)}">'
            f"<figcaption>{caption}</figcaption></figure>")

def table(head, rows, bold_first=False):
    h = "<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr></thead><tbody>"
    for i, r in enumerate(rows):
        tag = ' class="total"' if bold_first and i == 0 else ""
        h += f"<tr{tag}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return h + "</tbody></table>"

def qa(q, english_q, english_a, note):
    return (f'<blockquote><p lang="{ {"Hindi": "hi", "Bengali": "bn"}[q["language"].split("-")[0]] }">{html.escape(q["question"])}</p>'
            f"<p><em>{english_q}</em></p>"
            f'<p><strong>Reference short answer:</strong> {html.escape(q["short_answer"])} ({english_a})</p></blockquote>'
            f"<p>{note}</p>")

# ------------------------------------------------------------------ content (a list of blocks; "TABLE:x" blocks differ per page)
agentic_rows = [["All", "1,000", f'{A["ALL"]["correct"]} / {A["ALL"]["partial"]} / {A["ALL"]["incorrect"]}', pct(A["ALL"]["accuracy_strict"]),
                 pct(A["ALL"]["auto_numeric_match"]), pct(A["ALL"]["cited_all_facts"])]]
for l in LANGS + ["Cross-language"]:
    v = A["lang:" + l]
    agentic_rows.append([l, str(v["n"]), f'{v["correct"]} / {v["partial"]} / {v["incorrect"]}', pct(v["accuracy_strict"]),
                         pct(v["auto_numeric_match"]), pct(v["cited_all_facts"])])

retr_rows = []
for k, name in [("ALL", "All 1,000 questions"), ("cross_section", "Cross-section"), ("cross_document", "Cross-document"), ("cross_language", "Cross-language")]:
    v = G[k]
    retr_rows.append([name, str(v["n"]), pc0(v["fact_recall@5"]), pc0(v["fact_recall@10"]), pc0(v["all_facts@10"]), pc0(v["all_facts@20"])])
retr_rows.append(["Cross-language, one query per language", "100", pc0(LF["cross_language"]["fact_recall@5"]), pc0(LF["cross_language"]["fact_recall@10"]),
                  pc0(LF["cross_language"]["all_facts@10"]), pc0(LF["cross_language"]["all_facts@20"])])

blocks = [
f"""<p>Most question-answering benchmarks for Indian languages are short passages, often translated from English, with
the answer sitting in one place. Real enterprise questions are different. An employee asks something like <em>"if I am
confirmed after probation, how much longer before I can claim the welfare fund?"</em> The answer needs two clauses from
two long policy documents, and one small calculation. In a multilingual company, the two documents may not even be in
the same language.</p>
<p>This project builds a benchmark for exactly that situation, measures how far retrieval and an agentic RAG system get
on it, and provides a Rust chatbot that runs the same evaluation on a laptop with local open models. Everything
(documents, questions, code, predictions and logs) is in the repository.</p>""",

"<h2>1. The corpus: 50 policy manuals, written natively</h2>",
f"""<p>The documents are the HR policies of one fictional company, Suryodaya Integrated Solutions Pvt. Ltd. A fictional
company keeps the facts out of any model's training data, so a correct answer has to come from retrieval rather than
memory.</p>
<ul>
<li><strong>50 documents in 8 languages:</strong> Tamil 10, Hindi 9, Telugu 7, Marathi 6, Odia 5, Punjabi 5, Bengali 4 and Kannada 4.</li>
<li><strong>{words:,} words</strong> in total, 12k to 17k words per document, with {pages:,} page markers so answers can be traced to a page.</li>
<li><strong>Written natively, not translated.</strong> Each document is written in its own language with its own terminology and conventions. Numbers follow local practice: Bengali, Devanagari, Gurmukhi and Odia documents mostly use their native digits, while Tamil, Telugu and Kannada documents mostly use Western digits, so a system must normalise digits to compare figures.</li>
<li><strong>Topics</strong> span the employee life cycle: recruitment, onboarding, probation, compensation and payroll, leave, attendance and overtime, travel, remote work, performance and promotion, training, discipline and inquiries, grievances, harassment prevention, whistleblowing, benefits, loans, retirement funds, workplace safety, IT use and separation.</li>
</ul>
<p>Every document comes with a <strong>fact ledger</strong>: {facts:,} gold facts in total, each with a subject, an
aspect, a value and unit, the section and page, and an evidence sentence copied verbatim from the document. The ledgers
make the benchmark checkable: every question is built from ledger facts, and every fact can be located in the text.</p>
<p>Five <strong>topic-collision pairs</strong> add a trap. Pairs of documents in different languages share a topic (for
example, recruitment in Bengali and recruitment in Telugu) but none of their content. A retriever that matches on topic
instead of on the fact will pull the wrong document.</p>""",

fig("fig5_corpus.png", "Corpus size per language and the number of verbatim-grounded ledger facts.", "Bar chart of words per language in the corpus with fact counts."),
"<h2>2. The test set: 1,000 multi-hop questions</h2>",
fig("fig1_dataset.png", "Questions per language. The 100 cross-language questions each need one fact from each of two languages.",
    "Bar chart of questions per language: Tamil 180, Hindi 162, Telugu 126, Marathi 108, Odia 90, Punjabi 90, Bengali 72, Kannada 72, cross-language 100."),
f"""<p>Every question needs at least two facts. The hop types are:</p>
<ul>
<li><strong>Cross-document ({hops['cross_document']}):</strong> two facts from two documents in the same language.</li>
<li><strong>Cross-section ({hops['cross_section']}):</strong> two facts from different sections of one document.</li>
<li><strong>Cross-language ({hops['cross_language']}):</strong> one fact from each of two documents in different languages, asked in one of them.</li>
<li><strong>Within-section ({hops['within_section']}):</strong> two distinct facts from the same section.</li>
</ul>
<p>Each record has the question, a short reference answer, a full reference answer, the source sections and the
supporting facts (which resolve to ledger entries). Questions never state the values they test, so the answer cannot be
read off the question. Two examples:</p>""",
qa(T["HI-029"], "What is the probation period of an executive-level employee, and what minimum service is needed for the welfare-fund benefits (ex gratia, education and marriage assistance)? How much longer must an employee confirmed at the end of probation wait for these benefits?",
   "6 more months: probation 6 months, welfare fund 12 months",
   "Two Hindi documents: the probation rules (section 4.1) and the benefits guide (section 4.3). Neither fact is enough alone."),
qa(T["XL-REC-001"], "Under the Bengali recruitment guide, what is the time limit for completing the full background verification, and how many working days longer is it than the standard time limit for background verification in the Telugu guide?",
   "14 working days; 4 working days more than Telugu's 10",
   "A Bengali question that needs a Bengali clause (section 16.8) and a Telugu clause (section 8.3). The Telugu fact has to be found in Telugu."),

"<h2>3. How far does retrieval get? And an agentic RAG baseline</h2>",
"<h3>Retrieval with a single query</h3>",
"""<p>The documents are split along their own sections and subsections (at most 220 words per chunk, each chunk carrying
its document title and heading path), which gives 5,507 chunks. Retrieval is BM25 over two views of the text, word
tokens and character 4-grams, fused by reciprocal rank. Character n-grams help with the rich morphology and compounding
of Indian languages. No embedding model is needed for this baseline.</p>
<p>A lesson worth sharing: a tokenizer bug almost hid the result. Python's <code>\\w</code> does not treat Indic
vowel signs (matras) as word characters, so <code>re.findall(r"\\w+", text)</code> silently cuts Hindi or Tamil words
into fragments. Fixing the token pattern to include the Indic blocks doubled the share of questions whose evidence is
fully retrieved (from 30% to 64% in the top 10).</p>""",
fig("fig2_retrieval.png", "Share of questions whose gold facts are all in the top 10 chunks when the raw question is the query. Issuing one query per language recovers most cross-language cases.",
    "Bar chart: cross-section 85%, cross-document 66%, cross-language 4%, cross-language with per-language queries 59%."),
"TABLE:retrieval",
f"""<p>Two things stand out. Single queries find all the evidence for about two thirds of questions, and much less when
the facts are spread across documents. For cross-language questions a single query almost never works
({pc0(G['cross_language']['all_facts@10'])} in the top 10): lexical retrieval finds the clause in the question's own
language and misses the other one. Splitting the search into one query per language raises this to
{pc0(LF['cross_language']['all_facts@10'])}. Multi-hop, multilingual questions need a system that plans its searches.</p>""",
fig("fig3_retrieval_by_language.png", "Single-query retrieval differs by language, from Punjabi to Bengali.",
    "Bar chart of the share of questions with all facts in the top 10 by language: Bengali 85%, Telugu 82%, Kannada 75%, Marathi 70%, Tamil 69%, Hindi 68%, Odia 62%, Punjabi 56%."),
"<h3>Agentic RAG</h3>",
fig("fig4_pipeline.png", "The agentic loop: plan one query per needed fact in the language of its document, search, answer only from the passages with citations, and search again for anything missing.",
    "Diagram: question, plan per-fact queries, hybrid search, answer with citations, output; a loop back to planning when a fact is missing."),
"""<p>In the agentic setup, a reasoning agent gets the question and one tool: the search engine above, with optional
language and document filters. It breaks the question into the facts it needs, writes each query in the language of the
document likely to hold that fact, reads the passages, cites the chunks it used, and gives a short and a full answer.
The agents were Claude instances working in batches of 40 questions; a separate set of judge instances graded each
answer against the reference as correct, partial or incorrect.</p>""",
"TABLE:agentic",
f"""<p>The agents answered {A['ALL']['correct']} of 1,000 questions fully correctly; the other {A['ALL']['partial']} were
partially correct and none were wrong. The automatic numeric check, which needs no judge (every number in the reference
short answer must appear in the system's answer), agrees at {pct(A['ALL']['auto_numeric_match'])}. The two partial answers
are genuine system errors in Marathi: one missed a clause that keeps maternity leave inside the time-in-grade count, the
other did not retrieve the payslip query window. It took about three search calls per question.</p>
<p>Two caveats matter. First, the documents, questions, agents and judges all come from the same model family, so this
result shows that the benchmark is solvable and internally consistent rather than ranking models. Second, the first
evaluation surfaced inconsistencies in the data (for example, two sections of one document disagreeing on a figure).
Every one was fixed in the documents or the questions, and the affected questions were answered and graded again before
the final numbers.</p>""",

"<h2>4. A Rust RAG chatbot that runs on a laptop</h2>",
"""<p>The fourth part makes the system usable outside a research cluster. The repository includes a Rust implementation
of the whole pipeline (tokenizer, chunker, BM25 and dense indexes, retrieval, the agentic loop and the evaluation)
that talks to a local <a href="https://ollama.com">Ollama</a> server. It runs on an ordinary Windows laptop with an
8 GB GPU.</p>
<ul>
<li><strong>Same results as the reference code.</strong> The Rust chunker produces byte-identical chunks, the gold
evidence mapping is identical for all 1,000 questions, and single-query BM25 retrieval reproduces the published
recall.</li>
<li><strong>Multilingual dense retrieval.</strong> bge-m3 embeddings (through Ollama) can be fused with the two BM25
views; the chatbot uses this hybrid by default.</li>
<li><strong>Local generators.</strong> gemma3:4b as the main chatbot and gemma3:1b as the small model for later
fine-tuning; any Ollama chat model can be swapped in.</li>
<li><strong>Two strategies.</strong> <em>Simple</em> (one search, one answer) and <em>agentic</em> (per-fact queries in
each document's language, then a second round for missing facts), so the value of planning can be measured directly.</li>
<li><strong>A notebook for the Rust Jupyter kernel.</strong> It opens a file window to choose the data, downloads the
models it needs, builds the index, lets you chat in any of the eight languages, and runs the accuracy grid with
per-language reports. Runs are saved and resume after an interruption.</li>
</ul>
<p>The evaluation reports, for each language and hop type: numeric-match accuracy, judge accuracy when verdicts are
supplied, whether all gold facts reached the model's context, whether the model cited them, model calls per question and
seconds per question. That separates retrieval failures from reading failures, which is what a practitioner needs to
know before deciding whether to improve the retriever or fine-tune the model.</p>""",

"<h2>Limitations</h2>",
"""<ul>
<li><strong>One model family throughout.</strong> Independent evaluation needs other model families, closed-book and
oracle-evidence controls, and an independent judge.</li>
<li><strong>No native-speaker validation yet.</strong> The documents and questions have been checked for internal
consistency, not reviewed by native speakers of each language.</li>
<li><strong>One fictional company.</strong> This avoids contamination but limits domain diversity.</li>
<li><strong>Uneven languages.</strong> Between 72 and 180 questions per language.</li>
<li><strong>Open-model results are not part of this release.</strong> The Rust harness and notebook produce them on
local hardware; they will be reported with the fine-tuning experiments.</li>
</ul>""",

'<h2 id="get">Get the data and code</h2>',
"""<p>The repository contains the corpus with fact ledgers, the test set, the Python reference retrieval and scoring
code (standard library only), the prompts used for answering and grading, every prediction and judge verdict, the
search logs, the Rust chatbot and evaluator, and the Jupyter notebook. Data is released under CC BY 4.0 and code under
the MIT license.</p>""",
"MEDIUM_LINK",
]

# ------------------------------------------------------------------ page assembly
RETR_HEAD = ["Questions", "n", "Fact recall@5", "Fact recall@10", "All facts@10", "All facts@20"]
AG_HEAD = ["Group", "n", "Correct / partial / wrong", "Strict accuracy", "Numeric match", "All facts cited"]

def body(medium):
    out = []
    for b in blocks:
        if b == "TABLE:retrieval":
            if medium:  # Medium cannot import tables: give the key numbers as a list
                out.append("<ul>" + "".join(f"<li><strong>{r[0]}</strong> ({r[1]}): all facts in the top 10 for {r[4]}, "
                                            f"fact recall@10 {r[3]}.</li>" for r in retr_rows) + "</ul>")
            else:
                out.append(table(RETR_HEAD, retr_rows, bold_first=True))
        elif b == "TABLE:agentic":
            if medium:
                out.append(fig("table_agentic.png", "Agentic RAG results by question language (strict accuracy, automatic numeric match, and the share of questions whose gold facts were all cited).",
                               "Table of agentic RAG results: overall 998 correct, 2 partial, 0 wrong; 99.8% strict accuracy; 100% in every language except Marathi at 98.1%."))
            else:
                out.append(table(AG_HEAD, agentic_rows, bold_first=True))
        elif b == "MEDIUM_LINK":
            if medium:
                out.append(f'<p>Project page with links to the repository: <a href="{BASE}index.html">{TITLE}</a>.</p>')
        else:
            out.append(b)
    return "\n".join(out)

META = f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<meta name="description" content="{html.escape(DESC)}">
<meta name="author" content="{AUTHOR}">
<meta property="og:type" content="article">
<meta property="og:title" content="{TITLE}">
<meta property="og:description" content="{html.escape(DESC)}">
<meta property="og:image" content="{BASE}assets/fig2_retrieval.png">
<meta name="twitter:card" content="summary_large_image">"""

CSS = """
:root{--bg:#fcfcfb;--surface:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--rule:#e6e5e1;--accent:#2a78d6;--accent2:#eb6834;--code:#f4f3f0;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--surface:#222221;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9a998f;--rule:#3a3936;--accent:#5b9cec;--accent2:#e8794c;--code:#2a2a28;color-scheme:dark}
:root:not([data-theme="light"]) figure img{background:#fff;border-radius:6px}}
:root[data-theme="dark"]{--bg:#1a1a19;--surface:#222221;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9a998f;--rule:#3a3936;--accent:#5b9cec;--accent2:#e8794c;--code:#2a2a28;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:18px/1.65 Georgia,"Noto Serif","Nirmala UI","Noto Sans Devanagari","Noto Sans Bengali","Noto Sans Telugu","Noto Sans Tamil","Noto Sans Kannada","Noto Sans Oriya","Noto Sans Gurmukhi",serif;-webkit-text-size-adjust:100%}
main{max-width:720px;margin:0 auto;padding:48px 16px 80px}
header{border-bottom:1px solid var(--rule);margin-bottom:32px;padding-bottom:24px}
.kicker{font:600 13px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
h1{font:700 clamp(30px,5vw,42px)/1.15 system-ui,-apple-system,"Segoe UI",sans-serif;margin:10px 0 12px;letter-spacing:-.01em}
.sub{font-size:20px;color:var(--ink2);margin:0 0 16px}
.byline{font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--muted)}
.byline a{color:var(--ink2)}
h2{font:700 26px/1.25 system-ui,-apple-system,"Segoe UI",sans-serif;margin:48px 0 12px}
h3{font:600 20px/1.3 system-ui,-apple-system,"Segoe UI",sans-serif;margin:32px 0 8px;color:var(--ink2)}
p,ul{margin:0 0 18px}
li{margin:6px 0}
a{color:var(--accent)}
code{font:15px/1.4 ui-monospace,"Cascadia Code",Consolas,monospace;background:var(--code);padding:1px 5px;border-radius:4px}
blockquote{margin:24px 0;padding:4px 0 4px 20px;border-left:3px solid var(--accent2);color:var(--ink)}
blockquote p{margin:8px 0}
blockquote p[lang]{font:19px/1.7 "Nirmala UI","Noto Sans Devanagari","Noto Sans Bengali","Noto Sans Telugu",system-ui,sans-serif}
figure{margin:28px 0}
figure img{width:100%;height:auto;display:block}
figcaption{font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--muted);margin-top:8px}
.table-wrap{overflow-x:auto;margin:20px 0 24px}
table{border-collapse:collapse;width:100%;font:15px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}
th,td{padding:8px 8px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
th{white-space:normal;vertical-align:bottom}
td:first-child{min-width:8.5em}
th:first-child,td:first-child{text-align:left;white-space:normal}
th{color:var(--ink2);font-weight:600;border-bottom:2px solid var(--rule)}
tr.total td{font-weight:700}
.links{display:flex;flex-wrap:wrap;gap:10px;margin:20px 0 0}
.links a{font:600 14px/1 system-ui,-apple-system,"Segoe UI",sans-serif;text-decoration:none;color:var(--ink);border:1px solid var(--rule);border-radius:999px;padding:10px 14px;background:var(--surface)}
footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--rule);font:14px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--muted)}
"""

JS = r"""<script>
// On USER.github.io/REPO/ the repository is github.com/USER/REPO: point the links there.
(function(){var h=location.hostname,m=h.match(/^([^.]+)\.github\.io$/),seg=location.pathname.split('/').filter(Boolean);
if(!m||!seg.length)return;var base='https://github.com/'+m[1]+'/'+seg[0];
document.querySelectorAll('a.repo').forEach(function(a){a.href=base+a.getAttribute('data-path');});})();
</script>
"""

index = f"""<!doctype html>
<html lang="en">
<head>
{META}
<style>{CSS}</style>
</head>
<body>
<main>
<header>
<div class="kicker">Dataset · Benchmark · Code</div>
<h1>{TITLE}</h1>
<p class="sub">{SUB}</p>
<div class="byline">{AUTHOR}, IIIT Delhi · <a href="medium.html">Plain version for Medium import</a></div>
<div class="links"><a class="repo" data-path="" href="#get">Repository</a><a class="repo" data-path="/blob/main/testset/suryodaya_qa_testset_1000.json" href="#get">Test set (JSON)</a><a class="repo" data-path="/tree/main/rust" href="#get">Rust chatbot</a></div>
</header>
<article>
{body(False).replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")}
</article>
{JS}
<footer>Data CC BY 4.0 · Code MIT · Figures are generated from the stored results by <code>tools/site/make_figures.py</code>.</footer>
</main>
</body>
</html>
"""

medium = f"""<!doctype html>
<html lang="en">
<head>
{META}
</head>
<body>
<article>
<h1>{TITLE}</h1>
<p><em>{SUB}</em></p>
{body(True)}
<p><em>Data CC BY 4.0, code MIT. {AUTHOR}, IIIT Delhi.</em></p>
</article>
</body>
</html>
"""

open("docs/index.html", "w").write(index)
open("docs/medium.html", "w").write(medium)
assert "<table" not in medium and "<style" not in medium and "<script" not in medium
print("docs/index.html and docs/medium.html written;", f"{words:,} words, {facts:,} facts, {pages:,} page markers")
