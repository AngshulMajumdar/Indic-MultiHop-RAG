"""Validate one packet's output files: python3 -I validate.py <train_dir> <PACKET> [--final]
Mechanical checks only (no text is generated). --final also requires the quota to be met exactly."""
import json, glob, re, sys, collections, os
T, P = sys.argv[1], sys.argv[2]; final = "--final" in sys.argv
pk = json.load(open(f"{T}/packets/{P}.json"))
F = {f["fid"]: f for f in pk["facts"]}
REPO_CORPUS = os.path.join(T, "..", "repo", "suryodaya-indic-hr-mhqa", "corpus")
BL = {"Bengali": (0x980, 0x9FF), "Hindi": (0x900, 0x97F), "Marathi": (0x900, 0x97F), "Odia": (0xB00, 0xB7F),
      "Punjabi": (0xA00, 0xA7F), "Tamil": (0xB80, 0xBFF), "Telugu": (0xC00, 0xC7F), "Kannada": (0xC80, 0xCFF)}
DB = [0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6]
DMAP = {chr(b + i): str(i) for b in DB for i in range(10)}
def nums(s):
    s = "".join(DMAP.get(c, c) for c in str(s)).replace(",", "")
    return set(re.findall(r"\d+(?:\.\d+)?", s))
langs = pk.get("languages") or [pk["language"]]
allow = set()
for l in langs:
    for md in glob.glob(f"{REPO_CORPUS}/{l}/*.md"):
        allow |= set(re.findall(r"[A-Za-z][A-Za-z0-9&\-]*", open(md).read()))
def top(sec):
    return "".join(DMAP.get(c, c) for c in str(sec)).strip().split(".")[0].strip()
items = []
for fp in sorted(glob.glob(f"{T}/out/{P}_part*.json")):
    try:
        data = json.load(open(fp))
    except Exception as e:
        print(f"ERROR {os.path.basename(fp)}: invalid JSON: {e}"); sys.exit(1)
    for it in data:
        it["_file"] = os.path.basename(fp); items.append(it)
errs = collections.defaultdict(list); warns = collections.defaultdict(list)
seen_lid = collections.Counter(i.get("lid") for i in items)
use = collections.Counter(); pairs = collections.Counter(); qtext = collections.Counter()
hops = collections.Counter(); asked = collections.Counter()
for it in items:
    lid = it.get("lid", "?")
    E = errs[lid].append; W = warns[lid].append
    for k in ["lid", "hop_type", "asked_in", "fact_ids", "question", "short_answer", "answer"]:
        if k not in it or it[k] in ("", None, []): E(f"missing field {k}")
    if seen_lid[lid] > 1: E("duplicate lid")
    if not lid.startswith(P + "-"): E(f"lid must start with {P}-")
    fids = it.get("fact_ids") or []
    if len(fids) != 2 or len(set(fids)) != 2: E("need exactly 2 distinct fact_ids"); continue
    if any(f not in F for f in fids): E(f"unknown fact id {[f for f in fids if f not in F]}"); continue
    a, b = F[fids[0]], F[fids[1]]; h = it.get("hop_type")
    hops[h] += 1; asked[it.get("asked_in")] += 1
    for f in fids: use[f] += 1
    pairs[tuple(sorted(fids))] += 1
    if pk["kind"] == "mono":
        if it.get("asked_in") != pk["language"]: E("asked_in must be the packet language")
        if h == "cross_document" and a["document"] == b["document"]: E("cross_document needs two documents")
        if h == "cross_section" and (a["document"] != b["document"] or top(a["section"]) == top(b["section"])): E("cross_section needs one document, two different top-level sections")
        if h == "within_section" and (a["document"] != b["document"] or top(a["section"]) != top(b["section"])): E("within_section needs one document, same top-level section")
        if h not in ("cross_document", "cross_section", "within_section"): E(f"bad hop_type {h}")
    else:
        if h != "cross_language": E("hop_type must be cross_language")
        if a["language"] == b["language"]: E("cross_language needs one fact from each language")
        if it.get("asked_in") not in langs: E(f"asked_in must be one of {langs}")
    q = it.get("question", ""); L = it.get("asked_in") if it.get("asked_in") in BL else langs[0]
    lo, hi = BL[L]
    letters = [c for c in q if c.isalpha()]
    native = sum(lo <= ord(c) <= hi for c in letters)
    if letters and native / len(letters) < 0.6: E(f"question is not mainly in {L} script ({native}/{len(letters)} letters)")
    for fld in ("question", "short_answer", "answer"):
        bad = sorted(set(w for w in re.findall(r"[A-Za-z][A-Za-z0-9&\-]*", it.get(fld, "")) if w not in allow))
        if bad: E(f"English/Latin words not used in the documents, in {fld}: {bad[:8]}")
    qn = nums(q); sc = set(str(x) for x in it.get("scenario_numbers", []))
    for f in (a, b):
        leak = sorted(n for n in nums(f["value"]) if n in qn and n not in sc)
        if leak: E(f"question contains value number(s) {leak} of fact {f['fid']} (rewrite, or list in scenario_numbers if deliberately a scenario figure)")
    if len(it.get("short_answer", "").split()) > 20: W("short_answer longer than 20 words")
    qtext[q.strip()] += 1
for it in items:
    if qtext[it.get("question", "").strip()] > 1: errs[it.get("lid")].append("duplicate question text")
over = {f: n for f, n in use.items() if n > 2}
dup_pairs = [p for p, n in pairs.items() if n > 1]
nerr = sum(len(v) for v in errs.values())
for lid in sorted(errs):
    for e in errs[lid]: print(f"ERROR {lid}: {e}")
for lid in sorted(warns):
    for w in warns[lid]: print(f"warn  {lid}: {w}")
if over: print("ERROR facts used more than twice:", over); nerr += 1
if dup_pairs: print("ERROR same fact pair used twice:", dup_pairs); nerr += 1
q = {k: v for k, v in pk["quota"].items() if k != "asked_in"}
print(f"items {len(items)}; by hop {dict(hops)}; quota {q}" + (f"; asked_in {dict(asked)} target {pk['quota']['asked_in']}" if "asked_in" in pk["quota"] else ""))
if final:
    if dict(hops) != q: print("ERROR quota not met exactly"); nerr += 1
    if "asked_in" in pk["quota"] and dict(asked) != pk["quota"]["asked_in"]: print("ERROR asked_in split not met"); nerr += 1
print("OK" if nerr == 0 else f"{nerr} error(s)")
sys.exit(1 if nerr else 0)
