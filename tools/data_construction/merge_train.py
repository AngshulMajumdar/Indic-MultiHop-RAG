"""Assemble the training/dev sets from the authored packets and the verification verdicts (mechanical; no text generation).

usage: python3 -I tools/data_construction/merge_train.py <train_work_dir> <repo>
  <train_work_dir>/packets/<P>.json      facts given to each writer (only facts the test set does not use)
  <train_work_dir>/out/<P>_partNN.json   items written by hand by the writers
  <train_work_dir>/verify_out/<P>*.json  verdicts by separate reviewers: ok / fix (with corrected text) / drop
Writes <repo>/trainset/suryodaya_qa_train.json, suryodaya_qa_dev.json and stats.json.
"""
import collections, glob, hashlib, json, os, re, sys

W, REPO = sys.argv[1], sys.argv[2]
ACCEPT_UNVERIFIED = "--accept-unverified" in sys.argv
CODE = {"Bengali": "BN", "Hindi": "HI", "Marathi": "MR", "Odia": "OD", "Punjabi": "PA", "Tamil": "TA", "Telugu": "TE", "Kannada": "KN"}
test = json.load(open(f"{REPO}/testset/suryodaya_qa_testset_1000.json"))
test_facts = {(f["language"], f["document"], f["subject"], f["aspect"]) for q in test for f in q["supporting_facts"]}
test_q = {re.sub(r"\s+", " ", q["question"]).strip() for q in test}

verdict = {}
for fp in glob.glob(f"{W}/verify_out/*.json"):
    verdict.update(json.load(open(fp)))

items, stats = [], collections.Counter()
for pf in sorted(glob.glob(f"{W}/packets/*.json")):
    pk = json.load(open(pf)); P = pk["packet"]; F = {f["fid"]: f for f in pk["facts"]}
    for fp in sorted(glob.glob(f"{W}/out/{P}_part*.json")):
        for it in json.load(open(fp)):
            stats["written"] += 1
            v = verdict.get(it["lid"])
            if v is None:
                stats["unverified"] += 1
                if not ACCEPT_UNVERIFIED:
                    continue
                v = {"verdict": "unverified"}
            if v["verdict"] == "drop":
                stats["dropped"] += 1; continue
            if v["verdict"] == "fix":
                stats["fixed"] += 1
                for k in ("question", "short_answer", "answer"):
                    if v.get(k):
                        it[k] = v[k]
            elif v["verdict"] == "ok":
                stats["ok"] += 1
            fs = [F[x] for x in it["fact_ids"]]
            if pk["kind"] == "cross_language":  # the asked-in language first
                fs.sort(key=lambda f: f["language"] != it["asked_in"])
                lang = "-".join(f["language"] for f in fs)
            else:
                lang = pk["language"]
            for f in fs:
                assert (f["language"], f["document"], f["subject"], f["aspect"]) not in test_facts, f"test fact in {it['lid']}"
            assert re.sub(r"\s+", " ", it["question"]).strip() not in test_q, f"test question copied: {it['lid']}"
            items.append({"packet": P, "lid": it["lid"], "language": lang, "hop_type": it["hop_type"],
                          "question": it["question"].strip(), "short_answer": it["short_answer"].strip(), "answer": it["answer"].strip(),
                          "source": [{"language": f["language"], "document": f["document"], "section": str(f["section"])} for f in fs],
                          "supporting_facts": [{"language": f["language"], "document": f["document"], "subject": f["subject"], "aspect": f["aspect"]} for f in fs]})

# ids and a deterministic ~10% dev split
cnt = collections.Counter()
train, dev = [], []
for e in sorted(items, key=lambda e: (e["packet"], e["lid"])):
    if e["hop_type"] == "cross_language":
        tag = e["packet"][2:]; cnt["XL" + tag] += 1; e["id"] = f"TR-XL-{tag}-{cnt['XL' + tag]:03d}"
    else:
        c = CODE[e["language"]]; cnt[c] += 1; e["id"] = f"TR-{c}-{cnt[c]:04d}"
    e = {"id": e["id"], **{k: v for k, v in e.items() if k != "id"}}
    (dev if int(hashlib.md5(e["lid"].encode()).hexdigest(), 16) % 10 == 0 else train).append(e)

os.makedirs(f"{REPO}/trainset", exist_ok=True)
json.dump(train, open(f"{REPO}/trainset/suryodaya_qa_train.json", "w"), ensure_ascii=False, indent=1)
json.dump(dev, open(f"{REPO}/trainset/suryodaya_qa_dev.json", "w"), ensure_ascii=False, indent=1)
grp = lambda e: "Cross-language" if e["hop_type"] == "cross_language" else e["language"]
summary = {"verification": dict(stats), "train": len(train), "dev": len(dev),
           "train_by_group": dict(collections.Counter(grp(e) for e in train)),
           "train_by_hop": dict(collections.Counter(e["hop_type"] for e in train)),
           "facts_shared_with_test": 0}
json.dump(summary, open(f"{REPO}/trainset/stats.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
