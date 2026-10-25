"""Mechanical packet builder for the training set (no text generation).
Each packet: 2 documents (or a cross-language topic pair), a disjoint slice of facts NOT used by the test set, quotas."""
import json, glob, os, re, math, sys, collections
REPO = sys.argv[1]; OUT = sys.argv[2]
test = json.load(open(f"{REPO}/testset/suryodaya_qa_testset_1000.json"))
used = {(f["language"], f["document"], f["subject"], f["aspect"]) for q in test for f in q["supporting_facts"]}
CODE = {"Bengali": "BN", "Hindi": "HI", "Marathi": "MR", "Odia": "OD", "Punjabi": "PA", "Tamil": "TA", "Telugu": "TE", "Kannada": "KN"}
facts = collections.defaultdict(list); titles = {}
for p in sorted(glob.glob(f"{REPO}/ledgers/*/*.facts.json")):
    lang = p.split("/")[-2]; doc = os.path.basename(p)[:-11]
    md = open(p.replace("/ledgers/", "/corpus/")[:-11] + ".md").read()
    m = re.search(r"^##\s+(.+)$", md, re.M); titles[(lang, doc)] = m.group(1).strip() if m else doc
    for f in json.load(open(p)):
        if (lang, doc, f["subject"], f["aspect"]) in used:
            continue
        facts[(lang, doc)].append(f)
# test mix x2
mix = collections.Counter((q["language"], q["hop_type"]) for q in test if q["hop_type"] != "cross_language")
PER = 40
def split_quota(total, parts):
    base = [total // parts] * parts
    for i in range(total - sum(base)): base[i] += 1
    return base
def rr_pairs(docs):
    """round-robin schedule of document pairs (circle method)"""
    d = list(docs) + ([None] if len(docs) % 2 else [])
    n = len(d); out = []
    for r in range(n - 1):
        for i in range(n // 2):
            a, b = d[i], d[n - 1 - i]
            if a and b: out.append((a, b))
        d = [d[0]] + [d[-1]] + d[1:-1]
    return out
packets = []
for lang in CODE:
    docs = sorted(k[1] for k in facts if k[0] == lang)
    q = {h: 2 * mix[(lang, h)] for h in ["cross_document", "cross_section", "within_section"]}
    total = sum(q.values()); k = math.ceil(total / PER)
    pairs = rr_pairs(docs)[:k]
    use = collections.Counter(d for p in pairs for d in p)
    # disjoint contiguous slices of each document's facts, one per packet it appears in
    slices = {d: [facts[(lang, d)][i::use[d]] if False else None for i in range(use[d])] for d in docs if use[d]}
    for d in slices:
        fl = facts[(lang, d)]; n = use[d]; sz = math.ceil(len(fl) / n)
        slices[d] = [fl[i * sz:(i + 1) * sz] for i in range(n)]
    taken = collections.Counter()
    hq = {h: split_quota(v, k) for h, v in q.items()}
    for i, (a, b) in enumerate(pairs):
        pf = []
        for d in (a, b):
            for f in slices[d][taken[d]]:
                pf.append(dict(language=lang, document=d, doc_title=titles[(lang, d)], **f))
            taken[d] += 1
        packets.append({"packet": f"{CODE[lang]}{i+1:02d}", "kind": "mono", "language": lang, "documents": [a, b],
                        "quota": {h: hq[h][i] for h in hq if hq[h][i]}, "facts": pf})
# cross-language: the five topic pairs of the test set, 40 questions each (20 asked in each language)
xl = collections.OrderedDict()
for qq in test:
    if qq["hop_type"] == "cross_language":
        xl.setdefault(qq["id"].split("-")[1], tuple(sorted((s["language"], s["document"]) for s in qq["source"])))
for tag, pair in xl.items():
    pf = [dict(language=l, document=d, doc_title=titles[(l, d)], **f) for (l, d) in pair for f in facts[(l, d)]]
    packets.append({"packet": f"XL{tag}", "kind": "cross_language", "languages": [pair[0][0], pair[1][0]],
                    "documents": [f"{l}/{d}" for l, d in pair],
                    "quota": {"cross_language": 40, "asked_in": {pair[0][0]: 20, pair[1][0]: 20}}, "facts": pf})
# cap: at most 0.9 questions per available fact; move the excess (cross_document first) to the same language's richest packets
for lang in CODE:
    ps = [p for p in packets if p["kind"] == "mono" and p["language"] == lang]
    for p in ps:
        cap = int(0.9 * len(p["facts"])); excess = sum(p["quota"].values()) - cap
        while excess > 0:
            h = "cross_document" if p["quota"].get("cross_document", 0) > 0 else "cross_section"
            p["quota"][h] -= 1; excess -= 1
            tgt = max((x for x in ps if x is not p), key=lambda x: len(x["facts"]) - sum(x["quota"].values()))
            tgt["quota"][h] = tgt["quota"].get(h, 0) + 1
os.makedirs(OUT, exist_ok=True)
tot = collections.Counter()
for p in packets:
    for j, f in enumerate(p["facts"]):
        f["fid"] = f"{p['packet']}.f{j+1:03d}"
    json.dump(p, open(f"{OUT}/{p['packet']}.json", "w"), ensure_ascii=False, indent=1)
    n = sum(v for k, v in p["quota"].items() if k != "asked_in")
    tot[p["kind"]] += n
    print(p["packet"], p["documents"], p["quota"], "facts", len(p["facts"]))
print(len(packets), "packets", dict(tot), "questions", sum(tot.values()))
