"""Generate paper/generated/*.tex (numbers and tables) from the repository's data and result files.

Run from the repository root:  python3 paper/make_numbers.py
Fine-tuning results are read from results/finetune/<RUN>_results.json (written by notebooks/finetune_colab.ipynb);
until that file exists the fine-tuning table and sentences are typeset as pending.
"""
import collections, glob, json, os, re

R = "results"; OUT = "paper/generated"; RUN = os.environ.get("FT_RUN", "gemma3-1b")
os.makedirs(OUT, exist_ok=True)
LANGS = ["Bengali", "Hindi", "Kannada", "Marathi", "Odia", "Punjabi", "Tamil", "Telugu"]
macros = {}
def m(name, val):
    macros[name] = val
pct = lambda x: f"{100 * x:.1f}"

# ------------------------------------------------------------------ corpus and test set
docs = glob.glob("corpus/*/*.md")
words = sum(len(open(f).read().split()) for f in docs)
facts = sum(len(json.load(open(f))) for f in glob.glob("ledgers/*/*.facts.json"))
pages = sum(len(re.findall(r"<!--", open(f).read())) for f in docs)
test = json.load(open("testset/suryodaya_qa_testset_1000.json"))
hop = collections.Counter(q["hop_type"] for q in test)
used = {(f["language"], f["document"], f["subject"], f["aspect"]) for q in test for f in q["supporting_facts"]}
m("NDocs", str(len(docs))); m("NWords", f"{words:,}"); m("NWordsK", f"{words / 1000:.0f}k"); m("NFacts", f"{facts:,}")
m("NPages", f"{pages:,}"); m("NTest", f"{len(test):,}"); m("NTestFacts", f"{len(used):,}"); m("NFreeFacts", f"{facts - len(used):,}")
for h, n in hop.items():
    m("N" + h.replace("_", "").title().replace("Crossdocument", "CrossDoc").replace("Crosssection", "CrossSec")
      .replace("Crosslanguage", "CrossLang").replace("Withinsection", "WithinSec"), str(n))
perdoc = [len(open(f).read().split()) for f in docs]
m("MinDocWords", f"{min(perdoc) / 1000:.0f}k"); m("MaxDocWords", f"{max(perdoc) / 1000:.0f}k")

# dataset table
doc_n = collections.Counter(os.path.basename(os.path.dirname(f)) for f in docs)
q_n = collections.Counter(q["language"] for q in test if q["hop_type"] != "cross_language")
w_n = collections.Counter()
for f in docs:
    w_n[os.path.basename(os.path.dirname(f))] += len(open(f).read().split())
f_n = collections.Counter(os.path.basename(os.path.dirname(f)) for f in glob.glob("ledgers/*/*.facts.json") for _ in json.load(open(f)))
train = dev = None
if os.path.exists("trainset/suryodaya_qa_train.json"):
    train = json.load(open("trainset/suryodaya_qa_train.json")); dev = json.load(open("trainset/suryodaya_qa_dev.json"))
tr_n = collections.Counter(("Cross-language" if e["hop_type"] == "cross_language" else e["language"]) for e in (train or []) + (dev or []))
script = {"Bengali": "Bengali", "Hindi": "Devanagari", "Kannada": "Kannada", "Marathi": "Devanagari", "Odia": "Odia",
          "Punjabi": "Gurmukhi", "Tamil": "Tamil", "Telugu": "Telugu"}
rows = []
for l in sorted(LANGS, key=lambda l: -q_n[l]):
    rows.append(f"{l} & {script[l]} & {doc_n[l]} & {w_n[l] / 1000:.0f}k & {f_n[l]:,} & {q_n[l]} & {tr_n[l] if train else '--'} \\\\")
rows.append(r"\midrule")
rows.append(f"Cross-language & -- & -- & -- & -- & {hop['cross_language']} & {tr_n['Cross-language'] if train else '--'} \\\\")
rows.append(f"\\textbf{{Total}} & & {len(docs)} & {words / 1000:.0f}k & {facts:,} & {len(test):,} & {len(train) + len(dev) if train else '--'} \\\\")
open(f"{OUT}/table_dataset.tex", "w").write("\n".join(rows) + "\n")
if train:
    m("NTrain", f"{len(train):,}"); m("NDev", f"{len(dev):,}"); m("NTrainDev", f"{len(train) + len(dev):,}")
    tf = {(f["language"], f["document"], f["subject"], f["aspect"]) for e in train + dev for f in e["supporting_facts"]}
    m("NTrainFacts", f"{len(tf):,}"); m("NSharedFacts", str(len(tf & used)))
    st = json.load(open("trainset/stats.json"))["verification"]
    m("NWritten", f"{st.get('written', 0):,}"); m("NFixed", f"{st.get('fixed', 0):,}"); m("NDropped", f"{st.get('dropped', 0):,}")
    nver = st.get("ok", 0) + st.get("fixed", 0) + st.get("dropped", 0)
    if nver and not st.get("unverified"):
        m("TrainCheck", f"Training questions were written and verified under the same protocol: {st.get('written', 0):,} items were written, {st.get('fixed', 0):,} were corrected and {st.get('dropped', 0):,} were discarded by the independent verification pass.")
    elif nver:
        m("TrainCheck", f"Training questions were written under the same protocol and passed the same automatic checks; {nver:,} of them were additionally reviewed by independent verifiers, who corrected {st.get('fixed', 0):,} and discarded {st.get('dropped', 0):,}.")
    else:
        m("TrainCheck", "Training questions were written under the same protocol, each was re-checked by its writer against the verbatim evidence, and all passed the same automatic checks for fact presence, hop-type consistency, value leakage and script purity.")
else:
    for k in ["NTrain", "NDev", "NTrainDev", "NTrainFacts", "NWritten", "NFixed", "NDropped", "TrainCheck"]:
        m(k, r"\pending{}")
    m("NSharedFacts", "0")

# ------------------------------------------------------------------ retrieval
B = json.load(open(f"{R}/baseline_retrieval.json")); G, LF = B["global"], B["lang_filtered"]
rows = []
for key, name in [("ALL", r"\textbf{All}"), ("cross_section", "Cross-section"), ("cross_document", "Cross-document"),
                  ("within_section", "Within-section"), ("cross_language", "Cross-language")]:
    v = G[key]
    rows.append(f"{name} & {v['n']} & " + " & ".join(f"{v[k]:.3f}" for k in ["fact_recall@1", "fact_recall@5", "fact_recall@10", "all_facts@5", "all_facts@10", "all_facts@20"]) + r" \\")
v = LF["cross_language"]
rows.append(r"\midrule")
rows.append(f"Cross-language, per-language queries & {v['n']} & " + " & ".join(f"{v[k]:.3f}" for k in ["fact_recall@1", "fact_recall@5", "fact_recall@10", "all_facts@5", "all_facts@10", "all_facts@20"]) + r" \\")
open(f"{OUT}/table_retrieval.tex", "w").write("\n".join(rows) + "\n")
rows = []
for l in sorted(LANGS, key=lambda l: -G[l]["all_facts@10"]):
    v = G[l]
    rows.append(f"{l} & {v['n']} & {v['fact_recall@5']:.3f} & {v['fact_recall@10']:.3f} & {v['all_facts@10']:.3f} & {v['all_facts@20']:.3f} \\\\")
open(f"{OUT}/table_retrieval_lang.tex", "w").write("\n".join(rows) + "\n")
m("RAllTen", pct(G["ALL"]["all_facts@10"])); m("RFactTen", pct(G["ALL"]["fact_recall@10"]))
m("RXLTen", pct(G["cross_language"]["all_facts@10"])); m("RXLLFTen", pct(LF["cross_language"]["all_facts@10"]))
m("RCSTen", pct(G["cross_section"]["all_facts@10"])); m("RCDTen", pct(G["cross_document"]["all_facts@10"]))
best = max(LANGS, key=lambda l: G[l]["all_facts@10"]); worst = min(LANGS, key=lambda l: G[l]["all_facts@10"])
m("RBestLang", best); m("RBestLangV", pct(G[best]["all_facts@10"])); m("RWorstLang", worst); m("RWorstLangV", pct(G[worst]["all_facts@10"]))

# ------------------------------------------------------------------ agentic RAG
A = json.load(open(f"{R}/results_agentic_rag.json"))
rows = []
for key, name in [("ALL", r"\textbf{All}")] + [("lang:" + l, l) for l in LANGS] + [("lang:Cross-language", "Cross-language")]:
    v = A[key]
    rows.append(f"{name} & {v['n']} & {v['correct']}/{v['partial']}/{v['incorrect']} & {v['accuracy_strict']:.3f} & {v['accuracy_lenient']:.3f} & {v['auto_numeric_match']:.3f} & {v['cited_fact_recall']:.3f} & {v['cited_all_facts']:.3f} \\\\")
    if key == "ALL":
        rows.append(r"\midrule")
open(f"{OUT}/table_agentic.tex", "w").write("\n".join(rows) + "\n")
rows = []
for h in ["cross_document", "cross_section", "within_section", "cross_language"]:
    v = A["hop:" + h]
    rows.append(f"{h.replace('_', '-').capitalize()} & {v['n']} & {v['accuracy_strict']:.3f} & {v['auto_numeric_match']:.3f} & {v['cited_fact_recall']:.3f} & {v['cited_all_facts']:.3f} \\\\")
open(f"{OUT}/table_agentic_hop.tex", "w").write("\n".join(rows) + "\n")
m("AStrict", pct(A["ALL"]["accuracy_strict"])); m("ALenient", pct(A["ALL"]["accuracy_lenient"])); m("ANumeric", pct(A["ALL"]["auto_numeric_match"]))
m("ACitedRecall", pct(A["ALL"]["cited_fact_recall"])); m("ACitedAll", pct(A["ALL"]["cited_all_facts"]))
m("ACorrect", str(A["ALL"]["correct"])); m("APartial", str(A["ALL"]["partial"])); m("AIncorrect", str(A["ALL"]["incorrect"]))
m("ACalls", f"{A['_search_calls_total']:,}"); m("ACallsPerQ", f"{A['_search_calls_per_question']:.1f}")

# ------------------------------------------------------------------ fine-tuning (from the Colab notebook)
FT = f"{R}/finetune/{RUN}_results.json"
CSV = f"{R}/finetune/finetune_table.csv"
if not os.path.exists(FT) and os.path.exists(CSV):
    # hand-filled table -> the same structure the notebook writes
    import csv
    rows = [r for r in csv.reader(l for l in open(CSV) if not l.startswith("#"))]
    hdr, body = rows[0], rows[1:]
    tmp = collections.defaultdict(lambda: collections.defaultdict(dict)); filled = False
    for r in body:
        d = dict(zip(hdr, r))
        for c in ["closed", "rag", "oracle"]:
            for side, run in [("base", "base"), ("ft", "fine-tuned")]:
                v = d.get(f"{c}_{side}", "").strip()
                if v:
                    filled = True
                    tmp[f"{run} {c}"][d["group"]][d["metric"]] = float(v) / 100
                tmp[f"{run} {c}"][d["group"]]["n"] = int(d["n"])
    if filled:
        FT = CSV + ".json"
        json.dump(tmp, open(FT, "w"), indent=1)
groups = ["ALL"] + LANGS + ["Cross-language", "hop:cross_document", "hop:cross_section", "hop:within_section", "hop:cross_language"]
gname = lambda g: {"ALL": r"\textbf{All}"}.get(g, g.replace("hop:", "").replace("_", "-").capitalize() if g.startswith("hop:") else g)
conds = ["closed", "rag", "oracle"]
if os.path.exists(FT):
    F = json.load(open(FT))
    def val(run, g, metric="numeric_match"):
        x = F.get(run, {}).get(g, {}).get(metric)
        return None if x is None else 100 * x
    rows = []
    for g in groups:
        cells = []
        for c in conds:
            b, f = val(f"base {c}", g), val(f"fine-tuned {c}", g)
            cells += ["--" if b is None else f"{b:.1f}", "--" if f is None else (f"\\textbf{{{f:.1f}}}" if b is not None and f > b else f"{f:.1f}")]
        n = F.get("base rag", {}).get(g, {}).get("n", "")
        rows.append(f"{gname(g)} & {n} & " + " & ".join(cells) + r" \\")
        if g in ("ALL", "Cross-language"):
            rows.append(r"\midrule")
    open(f"{OUT}/table_finetune.tex", "w").write("\n".join(rows) + "\n")
    for c, C in [("closed", "Closed"), ("rag", "Rag"), ("oracle", "Oracle")]:
        b, f = val(f"base {c}", "ALL"), val(f"fine-tuned {c}", "ALL")
        m(f"FTBase{C}", f"{b:.1f}"); m(f"FTTuned{C}", f"{f:.1f}"); m(f"FTDelta{C}", f"{f - b:+.1f}")
        m(f"FTVerb{C}", "raises" if f > b else "lowers" if f < b else "leaves unchanged")
        bx, fx = val(f"base {c}", "Cross-language"), val(f"fine-tuned {c}", "Cross-language")
        m(f"FTXLBase{C}", f"{bx:.1f}"); m(f"FTXLTuned{C}", f"{fx:.1f}")
        se_b, se_f = val(f"base {c}", "ALL", "short_exact"), val(f"fine-tuned {c}", "ALL", "short_exact")
        m(f"FTSEBase{C}", f"{se_b:.1f}"); m(f"FTSETuned{C}", f"{se_f:.1f}")
    gic = val("base rag", "ALL", "gold_in_context"); m("FTGoldInCtxRag", f"{gic:.1f}")
    gains = {g: val("fine-tuned rag", g) - val("base rag", g) for g in LANGS if val("base rag", g) is not None}
    top = max(gains, key=gains.get); m("FTTopLang", top); m("FTTopLangDelta", f"{gains[top]:+.1f}")
    m("FTPending", "0")
else:
    rows = [f"{gname(g)} & & " + " & ".join([r"\pending{}"] * 6) + r" \\" + ("\n\\midrule" if g in ("ALL", "Cross-language") else "") for g in groups]
    open(f"{OUT}/table_finetune.tex", "w").write("\n".join(rows) + "\n")
    for C in ["Closed", "Rag", "Oracle"]:
        for k in ["FTBase", "FTTuned", "FTDelta", "FTXLBase", "FTXLTuned", "FTSEBase", "FTSETuned"]:
            m(k + C, r"\pending{}")
        m("FTVerb" + C, r"\pending{changes}")
    m("FTGoldInCtxRag", r"\pending{}"); m("FTTopLang", r"\pending{}"); m("FTTopLangDelta", r"\pending{}"); m("FTPending", "1")

with open(f"{OUT}/numbers.tex", "w") as f:
    f.write("% generated by paper/make_numbers.py — do not edit\n")
    for k, v in sorted(macros.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"{len(macros)} macros;", "fine-tuning results:", "found" if macros["FTPending"] == "0" else "pending")
