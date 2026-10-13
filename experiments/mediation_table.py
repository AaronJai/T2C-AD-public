"""Mediation analysis for Ch5 5.3.4: decompose each column's C3-C2 gain into
flips, and attribute flips to detector-flagged vs unflagged items.

Answers: 'is the C3-C2 lift the disambiguation stage, or C3's wider beam /
extra repair loop?'  Run:  python3 experiments/mediation_table.py
"""
import json, glob, os, collections

COLS = [
    ("M-ext",   "qual_records_mistral7b_pole_external_mistral.json"),
    ("C-v3",    "qual_records_claude-sonnet_v3_localdb.json"),
    ("C-ext",   "qual_records_claude-sonnet_pole_external_local.json"),
    ("Q32-v3",  "qual_records_qwen2.5-32b_v3_qwen.json"),
    ("Q32-ext", "qual_records_qwen2.5-32b_pole_external_qwen.json"),
    ("Q72-v3",  "qual_records_qwen2.5-72b_v3_qwen72.json"),
    ("Q72-ext", "qual_records_qwen2.5-72b_pole_external_qwen72.json"),
]
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

def load(fn):
    d = json.load(open(os.path.join(RES, fn)))
    idx = {}
    for cond in ("schema_grounded", "disambiguation_enhanced"):
        for r in d[cond]:
            idx.setdefault(r["question_id"], {})[cond] = r
    return idx

def flips(idx, metric):
    """metric: 'is_correct' (EX) or 'matches_any_interpretation' (EA)."""
    out = {"fixed": [], "regressed": []}
    for qid, pair in idx.items():
        c2, c3 = pair.get("schema_grounded"), pair.get("disambiguation_enhanced")
        if not c2 or not c3:
            continue
        a, b = bool(c2.get(metric)), bool(c3.get(metric))
        if not a and b:
            out["fixed"].append(c3)
        elif a and not b:
            out["regressed"].append(c3)
    return out

def amb(r):
    return bool(r.get("is_ambiguous_gold"))

print(f"{'col':8s} {'metric':4s} | {'fix':>3s} {'reg':>3s} | {'fix-amb':>7s} {'fix-unamb':>9s} | "
      f"{'flagged/fix-amb':>15s} | net-amb")
print("-" * 88)
summary = {}
for name, fn in COLS:
    idx = load(fn)
    row = {}
    for metric, tag in (("is_correct", "EX"), ("matches_any_interpretation", "EA")):
        f = flips(idx, metric)
        fa = [r for r in f["fixed"] if amb(r)]
        fu = [r for r in f["fixed"] if not amb(r)]
        ra = [r for r in f["regressed"] if amb(r)]
        flagged = [r for r in fa if r.get("ad_predicted_ambiguous") is True]
        row[tag] = dict(fixed=len(f["fixed"]), regressed=len(f["regressed"]),
                        fix_amb=len(fa), fix_unamb=len(fu), reg_amb=len(ra),
                        flagged=len(flagged))
        print(f"{name:8s} {tag:4s} | {len(f['fixed']):3d} {len(f['regressed']):3d} | "
              f"{len(fa):7d} {len(fu):9d} | {len(flagged):7d}/{len(fa):<7d} | "
              f"{len(fa)-len(ra):+d}")
    summary[name] = row

print("\n\n## Intent-type detection recall (gold-ambiguous intent items flagged by AD)")
for name, fn in COLS:
    idx = load(fn)
    tot = fl = 0
    for qid, pair in idx.items():
        c3 = pair.get("disambiguation_enhanced")
        if c3 and amb(c3) and c3.get("ambiguity_type") == "intent":
            tot += 1
            fl += 1 if c3.get("ad_predicted_ambiguous") is True else 0
    print(f"  {name:8s} intent flagged {fl:2d}/{tot:2d}"
          + (f"  ({100*fl/tot:.0f}%)" if tot else ""))

print("\n\n## Detector recall by type (all gold-ambiguous items)")
for name, fn in COLS:
    idx = load(fn)
    per = collections.defaultdict(lambda: [0, 0])
    for qid, pair in idx.items():
        c3 = pair.get("disambiguation_enhanced")
        if c3 and amb(c3):
            t = c3.get("ambiguity_type")
            per[t][0] += 1
            if c3.get("ad_predicted_ambiguous") is True:
                per[t][1] += 1
    parts = [f"{t} {v[1]}/{v[0]}" for t, v in sorted(per.items())]
    print(f"  {name:8s} " + "  ".join(parts))

print("\n\n## Per-type EA flips (fixed / regressed on matches_any), gold-ambiguous only")
for name, fn in COLS:
    idx = load(fn)
    per = collections.defaultdict(lambda: [0, 0, 0])   # fixed, regressed, fixed-and-flagged
    for qid, pair in idx.items():
        c2, c3 = pair.get("schema_grounded"), pair.get("disambiguation_enhanced")
        if not c2 or not c3 or not amb(c3):
            continue
        t = c3.get("ambiguity_type")
        a = bool(c2.get("matches_any_interpretation"))
        b = bool(c3.get("matches_any_interpretation"))
        if not a and b:
            per[t][0] += 1
            if c3.get("ad_predicted_ambiguous") is True:
                per[t][2] += 1
        elif a and not b:
            per[t][1] += 1
    parts = [f"{t}: +{v[0]}/-{v[1]} (flagged {v[2]})" for t, v in sorted(per.items())]
    print(f"  {name:8s} " + "  ".join(parts))
