"""Build verification work files: each item with the full text of its two facts (mechanical, no text generation).
python3 -I make_verify.py <train_dir> PACKET [PACKET ...]"""
import json, glob, sys, os
T = sys.argv[1]
for P in sys.argv[2:]:
    pk = json.load(open(f"{T}/packets/{P}.json")); F = {f["fid"]: f for f in pk["facts"]}
    items = [it for fp in sorted(glob.glob(f"{T}/out/{P}_part*.json")) for it in json.load(open(fp))]
    work = []
    for it in items:
        work.append({"lid": it["lid"], "hop_type": it["hop_type"], "asked_in": it["asked_in"], "question": it["question"],
                     "short_answer": it["short_answer"], "answer": it["answer"],
                     "facts": [{k: F[f][k] for k in ("fid", "language", "document", "doc_title", "section", "subject", "aspect", "value", "unit", "evidence") if k in F[f]} for f in it["fact_ids"]]})
    json.dump(work, open(f"{T}/verify_in/{P}.json", "w"), ensure_ascii=False, indent=1)
    print(P, len(work))
