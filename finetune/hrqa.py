"""Evaluation and LoRA fine-tuning of open models on the Suryodaya Indic HR multi-hop QA benchmark.

Pure Python (PyTorch + transformers + peft). Used by notebooks/finetune_colab.ipynb.

Conditions (each question is answered once per condition):
  closed  - no passages (contamination / prior-knowledge control)
  rag     - the top-k passages retrieved with the question (hybrid BM25, the repository's retriever)
  oracle  - the gold evidence passages plus retrieved distractors, shuffled (reading ability alone)

Metrics (per question language, cross-language, hop type):
  numeric match  - every number in the reference short answer appears in the model's short answer or answer
  short exact    - the set of numbers in the model's short answer equals the reference's (stricter)
  gold in context- whether every gold passage was in the prompt (rag/oracle)
"""
from __future__ import annotations

import glob, json, math, os, random, re, sys, time
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "rag"))
from ragcore import BM25, char_grams, chunk_doc, norm, word_tokens  # noqa: E402

LANGS = ["Bengali", "Hindi", "Kannada", "Marathi", "Odia", "Punjabi", "Tamil", "Telugu"]
DIGIT_BLOCKS = [0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6]
DMAP = {chr(b + i): str(i) for b in DIGIT_BLOCKS for i in range(10)}


# ----------------------------------------------------------------------------- data
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_test(repo=REPO):
    return load_json(os.path.join(repo, "testset", "suryodaya_qa_testset_1000.json"))


def load_train(repo=REPO):
    """Training and dev questions (facts disjoint from the test set)."""
    d = os.path.join(repo, "trainset")
    return load_json(os.path.join(d, "suryodaya_qa_train.json")), load_json(os.path.join(d, "suryodaya_qa_dev.json"))


def load_facts(repo=REPO):
    facts = {}
    for fp in glob.glob(os.path.join(repo, "ledgers", "*", "*.facts.json")):
        L = os.path.basename(os.path.dirname(fp)); D = os.path.basename(fp)[: -len(".facts.json")]
        for f in load_json(fp):
            facts[(L, D, f["subject"], f["aspect"])] = f
    return facts


class Corpus:
    """Section-aware chunks + hybrid BM25 (word unigrams and char 4-grams, reciprocal rank fusion), as in rag/."""

    def __init__(self, repo=REPO):
        self.chunks = []
        for md in sorted(glob.glob(os.path.join(repo, "corpus", "*", "*.md"))):
            lang = os.path.basename(os.path.dirname(md)); doc = os.path.basename(md)[:-3]
            if re.match(r"\d\d_", doc):
                self.chunks.extend(chunk_doc(lang, doc, md))
        texts = [c["doc_title"] + " " + c["heading"] + " " + c["text"] for c in self.chunks]
        self.word = BM25([word_tokens(t) for t in texts])
        self.char = BM25([char_grams(t) for t in texts])
        self.pos = {c["id"]: i for i, c in enumerate(self.chunks)}
        self.by_doc = defaultdict(list)
        for i, c in enumerate(self.chunks):
            self.by_doc[(c["language"], c["document"])].append(i)

    def search(self, q, lang=None, k=8):
        allowed = None if not lang else {i for i, c in enumerate(self.chunks) if c["language"] == lang}
        fused = {}
        for idx, toks in ((self.word, word_tokens(q)), (self.char, char_grams(q))):
            sc = idx.score(toks, allowed)
            for r, (i, _) in enumerate(sorted(sc.items(), key=lambda x: (-x[1], x[0]))[:200]):
                fused[i] = fused.get(i, 0) + 1 / (60 + r + 1)
        return [i for i, _ in sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:k]]

    def gold(self, items, facts):
        """Gold chunk ids per supporting fact (same procedure as eval/retrieval_baseline.py)."""
        clean = lambda s: re.sub(r"\s+", "", norm(s))
        ctext = [clean(c["text"]) for c in self.chunks]
        out = {}
        for e in items:
            g = []
            for s in e["supporting_facts"]:
                f = facts[(s["language"], s["document"], s["subject"], s["aspect"])]
                cand = self.by_doc[(s["language"], s["document"])]
                parts = [clean(p) for p in re.split(r"(?<=[।॥\.\?!])\s+", f["evidence"] or "") if len(p.strip()) > 8]
                ids = sorted({self.chunks[i]["id"] for i in cand for p in parts if p and p in ctext[i]})
                if not ids and cand:
                    ev = clean(f["evidence"] or "")
                    win = [ev[j:j + 40] for j in range(0, max(1, len(ev) - 40), 20)]
                    best = max(cand, key=lambda i: max((len(w) for w in win if w and w in ctext[i]), default=0))
                    if any(w and w in ctext[best] for w in win):
                        ids = [self.chunks[best]["id"]]
                g.append(ids)
            out[e["id"]] = g
        return out


def group_of(e):
    return "Cross-language" if e["hop_type"] == "cross_language" else e["language"]


def question_language(e):
    """Language the question is written in (for cross-language items, detected from the script)."""
    if e["hop_type"] != "cross_language":
        return e["language"]
    blocks = {"Bengali": (0x980, 0x9FF), "Hindi": (0x900, 0x97F), "Odia": (0xB00, 0xB7F), "Punjabi": (0xA00, 0xA7F),
              "Tamil": (0xB80, 0xBFF), "Telugu": (0xC00, 0xC7F), "Kannada": (0xC80, 0xCFF)}
    cands = e["language"].split("-")
    return max(cands, key=lambda l: sum(blocks.get(l, (0, -1))[0] <= ord(c) <= blocks.get(l, (0, -1))[1] for c in e["question"]))


# ----------------------------------------------------------------------------- prompts
INSTRUCTION = (
    "You are the HR-policy assistant of Suryodaya Integrated Solutions Pvt. Ltd. Answer the question using only the "
    "numbered passages from the company's HR documents (they may be in several Indian languages). The question may need "
    "facts from more than one passage: find each fact, then combine them and do exactly the comparison or arithmetic "
    "asked (keep working days and calendar days apart). Write in the language of the question. Reply in exactly this "
    "format:\nSHORT: <the minimal final answer: a number with its unit, yes/no with the decisive figure, a date or a short phrase>\n"
    "ANSWER: <2-4 sentences giving the facts you used, with their section numbers, and the reasoning>")
INSTRUCTION_CLOSED = (
    "You are the HR-policy assistant of Suryodaya Integrated Solutions Pvt. Ltd. Answer the question about the company's "
    "HR policies. Write in the language of the question. Reply in exactly this format:\n"
    "SHORT: <the minimal final answer>\nANSWER: <2-4 sentences with your reasoning>")


def passage_text(c, i, max_chars=1400):
    t = c["text"] if len(c["text"]) <= max_chars else c["text"][:max_chars] + " …"
    return f"[{i}] ({c['language']} | {c['doc_title']} | {c['heading']})\n{t}"


def contexts(corpus, items, gold, condition, k=6, seed=0):
    """Chunk indices to show for each item under a condition."""
    out = {}
    for e in items:
        if condition == "closed":
            out[e["id"]] = []
        elif condition == "rag":
            out[e["id"]] = corpus.search(e["question"], k=k)
        elif condition == "oracle":
            g = []
            for ids in gold[e["id"]]:
                for cid in ids[:1]:  # the first gold chunk of each fact
                    if corpus.pos[cid] not in g:
                        g.append(corpus.pos[cid])
            ds = [i for i in corpus.search(e["question"], k=k + len(g)) if i not in g][: max(0, k - len(g))]
            sel = g + ds
            random.Random(f"{seed}-{e['id']}").shuffle(sel)
            out[e["id"]] = sel
        else:
            raise ValueError(condition)
    return out


def messages(corpus, e, ctx_ids, answer=None):
    if ctx_ids:
        body = "\n\n".join(passage_text(corpus.chunks[i], n + 1) for n, i in enumerate(ctx_ids))
        user = f"{INSTRUCTION}\n\nPassages:\n\n{body}\n\nQuestion: {e['question']}"
    else:
        user = f"{INSTRUCTION_CLOSED}\n\nQuestion: {e['question']}"
    m = [{"role": "user", "content": user}]
    if answer is not None:
        m.append({"role": "assistant", "content": answer})
    return m


def target(e):
    return f"SHORT: {e['short_answer'].strip()}\nANSWER: {e['answer'].strip()}"


def parse(text):
    t = text.strip()
    m = re.search(r"SHORT\s*[:：]\s*(.*?)(?:\n\s*ANSWER\s*[:：]\s*(.*))?$", t, re.S | re.I)
    if not m:
        first = t.splitlines()[0] if t else ""
        return first.strip(), t
    return (m.group(1) or "").strip(), (m.group(2) or "").strip()


# ----------------------------------------------------------------------------- metrics
def numbers(s):
    s = "".join(DMAP.get(c, c) for c in str(s)).replace(",", "")
    out = []
    for x in re.findall(r"\d+(?:\.\d+)?", s):
        try:
            out.append(float(x))
        except ValueError:
            pass
    return out


def numeric_match(gold_short, short, answer):
    g = numbers(gold_short)
    if not g:
        return None
    p = numbers(short) + numbers(answer)
    return all(any(abs(x - y) < 1e-9 for y in p) for x in g)


def short_exact(gold_short, short):
    g = set(numbers(gold_short))
    if not g:
        return None
    return g == set(numbers(short))


def score(items, preds, gold=None):
    """preds: {id: {"short":..., "answer":..., "ctx": [chunk ids]}} -> {group: metrics}."""
    rows = defaultdict(Counter)
    for e in items:
        p = preds.get(e["id"])
        if p is None:
            continue
        nm = numeric_match(e["short_answer"], p.get("short", ""), p.get("answer", ""))
        se = short_exact(e["short_answer"], p.get("short", ""))
        ctx = set(p.get("ctx", []))
        g = [ids for ids in (gold or {}).get(e["id"], []) if ids]
        allin = all(any(i in ctx for i in ids) for ids in g) if g else None
        for key in ("ALL", group_of(e), "hop:" + e["hop_type"]):
            r = rows[key]
            r["n"] += 1
            if nm is not None:
                r["num_n"] += 1; r["num_ok"] += nm
            if se is not None:
                r["se_n"] += 1; r["se_ok"] += se
            if allin is not None and p.get("ctx"):
                r["ctx_n"] += 1; r["ctx_ok"] += allin
    out = {}
    for k, r in rows.items():
        out[k] = {"n": r["n"],
                  "numeric_match": r["num_ok"] / r["num_n"] if r["num_n"] else None,
                  "short_exact": r["se_ok"] / r["se_n"] if r["se_n"] else None,
                  "gold_in_context": r["ctx_ok"] / r["ctx_n"] if r["ctx_n"] else None}
    return out


def order_groups(keys):
    return sorted(keys, key=lambda k: (k != "ALL", k.startswith("hop:"), k == "Cross-language", k))


def table(results, metric="numeric_match"):
    """results: {run name: score()} -> markdown table of one metric by group."""
    groups = order_groups({g for r in results.values() for g in r})
    lines = ["| group | " + " | ".join(results) + " |", "|---|" + "---|" * len(results)]
    for g in groups:
        cells = []
        for r in results.values():
            v = r.get(g, {}).get(metric)
            cells.append("-" if v is None else f"{100 * v:.1f}")
        n = next((r[g]["n"] for r in results.values() if g in r), "")
        lines.append(f"| {g} (n={n}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- generation
def load_model(name, adapter=None, dtype=None, device=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).to(device)
    model.eval()
    return model, tok


def _prompt_text(tok, msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate(model, tok, prompts, max_new_tokens=256, batch_size=8):
    import torch
    outs = []
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))  # similar lengths per batch
    res = [None] * len(prompts)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            res[i] = tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return res


def evaluate(model, tok, corpus, items, gold, condition, out_path, k=6, batch_size=8, max_new_tokens=256, log_every=50):
    """Answer every item under `condition`; appends to out_path (JSONL) and resumes. Returns {id: prediction}."""
    done = {}
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            if line.strip():
                p = json.loads(line); done[p["id"]] = p
    todo = [e for e in items if e["id"] not in done]
    ctx = contexts(corpus, todo, gold, condition, k=k)
    t0 = time.time()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        step = max(batch_size, log_every)
        for s in range(0, len(todo), step):
            part = todo[s:s + step]
            prompts = [_prompt_text(tok, messages(corpus, e, ctx[e["id"]])) for e in part]
            texts = generate(model, tok, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
            for e, t in zip(part, texts):
                short, ans = parse(t)
                p = {"id": e["id"], "condition": condition, "short": short, "answer": ans, "raw": t,
                     "ctx": [corpus.chunks[i]["id"] for i in ctx[e["id"]]]}
                done[e["id"]] = p
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
            f.flush()
            n = s + len(part); el = time.time() - t0
            print(f"  {condition}: {n}/{len(todo)}  {el / n:.2f} s/q  ~{el / n * (len(todo) - n) / 60:.0f} min left", flush=True)
    return {e["id"]: done[e["id"]] for e in items if e["id"] in done}


# ----------------------------------------------------------------------------- training
def training_examples(corpus, items, gold, tok=None, k=6, max_len=4096, seed=0):
    """RAFT-style examples: gold passages + retrieved hard distractors, shuffled; target = reference answer.
    With a tokenizer, distractors are dropped (never gold passages) until the example fits max_len tokens."""
    ctx = contexts(corpus, items, gold, "oracle", k=k, seed=seed)
    out = []
    for e in items:
        ids = list(ctx[e["id"]])
        goldpos = {corpus.pos[c] for g in gold[e["id"]] for c in g[:1]}
        while True:
            ex = {"id": e["id"], "messages": messages(corpus, e, ids, answer=target(e))}
            if tok is None or len(_encode(tok, ex, 10 ** 9)[0]) <= max_len or not any(i not in goldpos for i in ids):
                break
            drop = max(i for i, x in enumerate(ids) if x not in goldpos)
            ids.pop(drop)
        out.append(ex)
    return out


def _encode(tok, ex, max_len):
    prompt = tok.apply_chat_template(ex["messages"][:-1], tokenize=False, add_generation_prompt=True)
    full = prompt + ex["messages"][-1]["content"] + "<end_of_turn>\n" if "<end_of_turn>" in prompt else \
        prompt + ex["messages"][-1]["content"] + (tok.eos_token or "")
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    p_len = len(p_ids)
    if len(f_ids) > max_len:  # last resort: cut the middle of the prompt, keep its start, the question and the answer
        cut = len(f_ids) - max_len
        keep = 64
        f_ids = f_ids[:keep] + f_ids[keep + cut:]
        p_len = max(keep, p_len - cut)
    labels = [-100] * p_len + f_ids[p_len:]
    return f_ids, labels


def train_lora(model_name, examples, out_dir, epochs=2, lr=2e-4, r=16, alpha=32, dropout=0.05, max_len=4096,
               grad_accum=8, warmup=0.05, seed=0, log_every=10, dev_examples=None, target_modules=None):
    """Minimal, version-stable LoRA training loop (loss on the answer tokens only). Saves the adapter to out_dir."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(seed); random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if bf16 else torch.float32,
                                                 attn_implementation="eager").to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, task_type="CAUSAL_LM",
                     target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    data = [_encode(tok, ex, max_len) for ex in examples]
    dev = [_encode(tok, ex, max_len) for ex in (dev_examples or [])]
    print(f"{len(data)} training examples; tokens per example: mean {sum(len(d[0]) for d in data) / max(1, len(data)):.0f}, "
          f"max {max((len(d[0]) for d in data), default=0)}")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    total = math.ceil(len(data) * epochs / grad_accum)
    warm = max(1, int(warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    use_fp16 = device == "cuda" and not bf16
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    def loss_of(ids, labels):
        x = torch.tensor([ids], device=device); y = torch.tensor([labels], device=device)
        if use_fp16:
            with torch.autocast("cuda", dtype=torch.float16):
                return model(input_ids=x, labels=y).loss
        return model(input_ids=x, labels=y).loss

    def dev_loss():
        if not dev:
            return None
        model.eval(); tot = 0.0
        with torch.no_grad():
            for ids, labels in dev:
                tot += float(loss_of(ids, labels))
        model.train()
        return tot / len(dev)

    model.train(); step = 0; t0 = time.time(); run = 0.0; history = []
    for ep in range(epochs):
        order = list(range(len(data))); random.shuffle(order)
        for n, i in enumerate(order):
            ids, labels = data[i]
            loss = loss_of(ids, labels) / grad_accum
            (scaler.scale(loss) if scaler else loss).backward()
            run += float(loss) * grad_accum
            if (n + 1) % grad_accum == 0 or n + 1 == len(order):
                if scaler:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                (scaler.step(opt) if scaler else opt.step())
                if scaler:
                    scaler.update()
                opt.zero_grad(set_to_none=True); sched.step(); step += 1
                if step % log_every == 0 or step == total:
                    el = time.time() - t0
                    print(f"epoch {ep + 1} step {step}/{total} loss {run / (grad_accum * log_every):.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} {el / step:.1f} s/step ~{el / step * (total - step) / 60:.0f} min left", flush=True)
                    history.append({"step": step, "loss": run / (grad_accum * log_every)}); run = 0.0
        dl = dev_loss()
        if dl is not None:
            print(f"epoch {ep + 1}: dev loss {dl:.4f}", flush=True)
            history.append({"epoch": ep + 1, "dev_loss": dl})
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    json.dump({"base_model": model_name, "epochs": epochs, "lr": lr, "r": r, "alpha": alpha, "max_len": max_len,
               "grad_accum": grad_accum, "examples": len(data), "history": history},
              open(os.path.join(out_dir, "train_log.json"), "w"), indent=1)
    print("adapter saved to", out_dir)
    return out_dir
