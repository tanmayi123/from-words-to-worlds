"""The human check of the judges' labels: agreement between people, and each labelling rule against them."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

d = Path(sys.argv[1])
s = pd.read_csv(d / "audit_sample.csv").set_index("item")
shares = json.load(open(d / "group_shares.json"))
ans = {}
for n in ("santhosh", "praniti", "tanmayi", "mukul"):
    t = open(d / f"answers_{n}.txt").read().split()[2:]
    ans[n] = pd.Series({"p" + x[:3]: x[3] for x in t})
    print(n, len(ans[n]), ans[n].value_counts().to_dict())
long = pd.concat([a.rename("ans").to_frame().assign(rater=n) for n, a in ans.items()]).rename_axis("item").reset_index()
per = long.groupby("item").ans.apply(list)
assert (per.map(len) == 2).all(), "every picture needs two answers"

# people against people
a = np.array([p[0] for p in per]); b = np.array([p[1] for p in per])
both = (a != "u") & (b != "u")
po = (a == b).mean()
cats = ["y", "n", "u"]
pe = sum((a == c).mean() * (b == c).mean() for c in cats)
dec_agree = (a[both] == b[both]).mean()
pa, pb = (a[both] == "y").mean(), (b[both] == "y").mean()
kappa_dec = ((a[both] == b[both]).mean() - (pa * pb + (1 - pa) * (1 - pb))) / (1 - (pa * pb + (1 - pa) * (1 - pb)))
print(f"\nPEOPLE: exact agreement {po:.3f}, kappa (3 answers) {(po - pe) / (1 - pe):.3f}; "
      f"both decided on {both.sum()} of {len(a)}: agree {dec_agree:.3f}, kappa {kappa_dec:.3f}")


def verdict(p):
    dec = [x for x in p if x != "u"]
    if not dec:
        return "unclear"
    if all(x == "y" for x in dec):
        return "present"
    if all(x == "n" for x in dec):
        return "missing"
    return "conflict"


s["human"] = per.map(verdict)
print("\nhuman verdict by judge group:")
print(pd.crosstab(s.group, s.human, margins=True))

v = s[s.human.isin(["present", "missing"])].copy()
v["truth_missing"] = (v.human == "missing").astype(int)
v["w"] = v.group.map(lambda g: shares[g]) / v.groupby("group").group.transform("size")
rules = {"our rule: missing when both miss (FRCNN >= 0.5 or OWL >= 0.1)": ~((v.frcnn >= 0.5) | (v.owl >= 0.1)),
         "stricter: missing unless both find": ~((v.frcnn >= 0.5) & (v.owl >= 0.1)),
         "Faster R-CNN alone": v.frcnn < 0.5, "OWL-ViT alone": v.owl < 0.1, "RetinaNet alone (0.5)": v.retina < 0.5}
print(f"\nRULES against people, on {len(v)} pictures with a clear human verdict, weighted to all development objects:")
rows = []
for name, pred in rules.items():
    pred = pred.astype(int).to_numpy(); t = v.truth_missing.to_numpy(); w = v.w.to_numpy()
    acc = (w * (pred == t)).sum() / w.sum()
    sens = (w * ((pred == 1) & (t == 1))).sum() / (w * (t == 1)).sum()     # real misses caught
    ppv = (w * ((pred == 1) & (t == 1))).sum() / max((w * (pred == 1)).sum(), 1e-12)  # called missing that really are
    rows.append({"rule": name, "accuracy": round(acc, 3), "real misses caught": round(sens, 3),
                 "called missing that really are": round(ppv, 3)})
print(pd.DataFrame(rows).to_string(index=False))
rng = np.random.default_rng(0)
t = v.truth_missing.to_numpy(); w = v.w.to_numpy(); pred = rules[list(rules)[0]].astype(int).to_numpy()
boots = []
for _ in range(2000):
    i = rng.integers(0, len(v), len(v))
    boots.append((w[i] * (pred[i] == t[i])).sum() / w[i].sum())
print("our rule accuracy 95% interval:", np.round(np.percentile(boots, [2.5, 97.5]), 3))


def auroc(score, y, w):
    pos, neg = y == 1, y == 0
    tot = 0.0
    for i in np.where(pos)[0]:
        tot += w[i] * ((w[neg] * (score[neg] < score[i])).sum() + 0.5 * (w[neg] * (score[neg] == score[i])).sum())
    return tot / (w[pos].sum() * w[neg].sum())


print("\nCEILING against people (weighted):")
for col in ("retina", "frcnn", "owl"):
    print(f"  {col} score on the finished picture: AUROC {auroc(-v[col].to_numpy(), t, w):.3f}")
lab = (~((v.frcnn >= 0.5) | (v.owl >= 0.1))).astype(int).to_numpy()
print(f"  our labels themselves as a predictor of the human verdict: AUROC {auroc(lab.astype(float), t, w):.3f}")
s.reset_index()[["item", "seed_id", "caption_idx", "category", "word_start", "group", "frcnn", "owl", "retina", "human"]].to_csv(
    d / "audit_verdicts.csv", index=False)
