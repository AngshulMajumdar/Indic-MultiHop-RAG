"""Builds notebooks/finetune_colab.ipynb (Python; Google Colab). Run from the repository root."""
import json

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# Fine-tuning a small open model for multilingual multi-hop HR QA

Runs on Google Colab with a GPU. Everything is resumable: answers and the trained adapter are saved to your Google
Drive, so if Colab disconnects, re-run the cells and they continue where they stopped.

1. **Baseline:** the base model answers all 1,000 test questions under three conditions:
   *closed-book* (no passages), *RAG* (passages retrieved with the question) and *oracle* (the gold passages plus
   retrieved distractors).
2. **Fine-tuning:** LoRA on the training questions (their facts never appear in the test set), with the gold passages
   mixed with hard distractors, so the model learns to find and combine the right facts.
3. **Re-evaluation** under the same three conditions, and before/after tables by language and hop type.

**Runtime:** *Runtime → Change runtime type → GPU*. An L4 or A100 is fastest (bfloat16); a T4 works but is slower.
""")
md("## 1. Get the repository")
code("""
import os, subprocess, sys, glob
# Option A: a public GitHub repository (fill in), Option B: upload the repository zip when asked.
REPO_URL = ""   # e.g. "https://github.com/<user>/suryodaya-indic-hr-mhqa"
WORK = "/content"
if REPO_URL:
    if not os.path.isdir(f"{WORK}/repo"):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, f"{WORK}/repo"], check=True)
    REPO = f"{WORK}/repo"
else:
    found = glob.glob(f"{WORK}/**/testset/suryodaya_qa_testset_1000.json", recursive=True)
    if not found:
        from google.colab import files
        print("Choose the repository zip (suryodaya-indic-hr-mhqa.zip):")
        up = files.upload()
        z = next(iter(up))
        subprocess.run(["unzip", "-q", "-o", z, "-d", WORK], check=True)
        found = glob.glob(f"{WORK}/**/testset/suryodaya_qa_testset_1000.json", recursive=True)
    REPO = os.path.dirname(os.path.dirname(found[0]))
print("repository:", REPO)
""")
code("""
# Results go to Google Drive so they survive a disconnect.
from google.colab import drive
drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/suryodaya_finetune"
os.makedirs(OUT, exist_ok=True)
print("results folder:", OUT)
""")
code("""
!pip -q install -U "transformers>=4.50" "peft>=0.13" accelerate
import torch
print(torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
      "| bf16:", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
""")
md("""
## 2. Settings

`BASE_MODEL` must have a chat template. `unsloth/gemma-3-1b-it` is an ungated copy of Google's Gemma 3 1B instruct
model (no Hugging Face login needed); `google/gemma-3-1b-it` is the same model behind a licence click-through.
""")
code("""
BASE_MODEL = "unsloth/gemma-3-1b-it"
RUN = "gemma3-1b"                  # name used for the result files
CONDITIONS = ["closed", "rag", "oracle"]
K = 6                              # passages per question (rag: top-6 retrieved; oracle: gold + distractors to 6)
MAX_NEW_TOKENS = 320
EPOCHS, LR, LORA_R, LORA_ALPHA, MAX_LEN, GRAD_ACCUM = 2, 2e-4, 16, 32, 4096, 8
gpu_gb = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
BATCH = 16 if gpu_gb >= 39 else 8 if gpu_gb >= 20 else 4
print("eval batch size", BATCH)
""")
md("## 3. Data, retrieval index and leakage check")
code("""
sys.path.insert(0, os.path.join(REPO, "finetune"))
import importlib, json, hrqa as H
importlib.reload(H)
corpus = H.Corpus(REPO)
facts = H.load_facts(REPO)
test = H.load_test(REPO)
train, dev = H.load_train(REPO)
gold_test = corpus.gold(test, facts)
gold_train = corpus.gold(train + dev, facts)
key = lambda f: (f["language"], f["document"], f["subject"], f["aspect"])
test_facts = {key(f) for e in test for f in e["supporting_facts"]}
train_facts = {key(f) for e in train + dev for f in e["supporting_facts"]}
print(f"{len(corpus.chunks)} chunks | test {len(test)} | train {len(train)} | dev {len(dev)}")
print("facts shared by train/dev and test:", len(test_facts & train_facts))
assert not (test_facts & train_facts), "leakage: a training fact is used by the test set"
""")
md("## 4. Baseline: the base model on all 1,000 test questions")
code("""
model, tok = H.load_model(BASE_MODEL)
results = {}
for cond in CONDITIONS:
    preds = H.evaluate(model, tok, corpus, test, gold_test, cond, f"{OUT}/{RUN}_base_{cond}.jsonl",
                       k=K, batch_size=BATCH, max_new_tokens=MAX_NEW_TOKENS)
    results[f"base {cond}"] = H.score(test, preds, gold_test)
print(H.table(results, "numeric_match"))
""")
md("## 5. LoRA fine-tuning")
code("""
del model; torch.cuda.empty_cache()
ADAPTER = f"{OUT}/{RUN}_lora"
ex_train = H.training_examples(corpus, train, gold_train, tok=tok, k=K, max_len=MAX_LEN)
ex_dev = H.training_examples(corpus, dev, gold_train, tok=tok, k=K, max_len=MAX_LEN)
if not os.path.exists(os.path.join(ADAPTER, "adapter_config.json")):
    H.train_lora(BASE_MODEL, ex_train, ADAPTER, epochs=EPOCHS, lr=LR, r=LORA_R, alpha=LORA_ALPHA,
                 max_len=MAX_LEN, grad_accum=GRAD_ACCUM, dev_examples=ex_dev)
else:
    print("adapter already trained:", ADAPTER)
print(json.load(open(os.path.join(ADAPTER, "train_log.json")))["history"][-3:])
""")
md("## 6. The fine-tuned model on the same 1,000 questions")
code("""
model, tok = H.load_model(BASE_MODEL, adapter=ADAPTER)
for cond in CONDITIONS:
    preds = H.evaluate(model, tok, corpus, test, gold_test, cond, f"{OUT}/{RUN}_ft_{cond}.jsonl",
                       k=K, batch_size=BATCH, max_new_tokens=MAX_NEW_TOKENS)
    results[f"fine-tuned {cond}"] = H.score(test, preds, gold_test)
""")
md("""
## 7. Results

*numeric match*: every number in the reference short answer appears in the model's answer. *short exact*: the
numbers in the model's short answer are exactly the reference's. *gold in context*: the passages shown contained
every gold fact (identical for base and fine-tuned, because retrieval does not change).
""")
code("""
from IPython.display import Markdown, display
order = [f"{s} {c}" for c in CONDITIONS for s in ("base", "fine-tuned")]
R = {k: results[k] for k in order if k in results}
for metric in ["numeric_match", "short_exact", "gold_in_context"]:
    display(Markdown(f"### {metric}\\n\\n" + H.table(R, metric)))
json.dump(R, open(f"{OUT}/{RUN}_results.json", "w"), indent=1)
open(f"{OUT}/{RUN}_results.md", "w").write("\\n\\n".join(f"## {m}\\n\\n" + H.table(R, m) for m in ["numeric_match", "short_exact", "gold_in_context"]))
print("saved", f"{OUT}/{RUN}_results.json", "and .md")
""")
code("""
import matplotlib.pyplot as plt
groups = ["ALL", "Bengali", "Hindi", "Kannada", "Marathi", "Odia", "Punjabi", "Tamil", "Telugu", "Cross-language"]
fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(5 * len(CONDITIONS), 4.5), sharey=True)
for ax, c in zip(axes, CONDITIONS):
    b = [100 * (R[f"base {c}"].get(g, {}).get("numeric_match") or 0) for g in groups]
    f = [100 * (R[f"fine-tuned {c}"].get(g, {}).get("numeric_match") or 0) for g in groups]
    y = range(len(groups))
    ax.barh([i + 0.2 for i in y], b, height=0.4, color="#9ec5f0", label="base")
    ax.barh([i - 0.2 for i in y], f, height=0.4, color="#2a78d6", label="fine-tuned")
    ax.set_yticks(list(y)); ax.set_yticklabels(groups); ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.set_title(c); ax.set_xlabel("numeric match %")
axes[0].legend(loc="lower right")
plt.tight_layout(); plt.savefig(f"{OUT}/{RUN}_before_after.png", dpi=160); plt.show()
""")
md("""
## 8. Optional: use the fine-tuned model in Ollama

Merge the adapter and save a full model folder; then on your computer: convert with llama.cpp's
`convert_hf_to_gguf.py`, and `ollama create hr-gemma3-1b -f Modelfile` (Modelfile: `FROM ./model.gguf`).
""")
code("""
MERGED = f"{OUT}/{RUN}_merged"
if not os.path.isdir(MERGED):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
    merged.save_pretrained(MERGED); AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(MERGED)
print("merged model:", MERGED)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
      "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"c{i:02d}"
json.dump(nb, open("notebooks/finetune_colab.ipynb", "w"), ensure_ascii=False, indent=1)
print(len(cells), "cells -> notebooks/finetune_colab.ipynb")
