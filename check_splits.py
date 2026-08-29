#!/usr/bin/env python3
"""check_splits.py — verify the splits before spending money on training."""
import csv, os, sys, collections

d = sys.argv[1] if len(sys.argv) > 1 else "splits"
base = sys.argv[2] if len(sys.argv) > 2 else "baseline.csv"

S, T = {}, {}
for n in ("train", "val", "test"):
    rows = list(csv.DictReader(open(os.path.join(d, f"{n}.csv"), encoding="utf-8")))
    S[n] = {r["image_path"] for r in rows}
    T[n] = rows
    print(f"{n:<6}{len(rows)}")

ok = True
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    n = len(S[a] & S[b])
    print(f"overlap {a}/{b}: {n}")
    ok &= n == 0

if os.path.exists(base):
    prot = {r["image_path"] for r in csv.DictReader(open(base, encoding="utf-8"))
            if r.get("image_path")}
    miss = prot - S["test"]
    leaked = prot & (S["train"] | S["val"])
    print(f"baselined images in test: {len(prot & S['test'])}/{len(prot)}")
    print(f"baselined images leaked into train/val: {len(leaked)}")
    ok &= not leaked

STOP = {"THE","OF","AND","A","IN","TO","MY","IS","ON","FOR","AT","DE","LA","EL",
        "DER","DIE","DAS","NO","UN"}
tok = lambda s: {t for t in s.upper().split() if len(t) >= 4 and t not in STOP}
tt = set()
for r in T["test"]:
    tt |= tok(r["target_text"])
bad = [r["target_text"] for r in T["train"] if tok(r["target_text"]) & tt]
print(f"train names sharing a test token: {len(bad)}")
ok &= not bad

missing = [r["image_path"] for n in S for r in T[n]
           if not os.path.exists(r["image_path"])]
print(f"missing image files: {len(missing)}")
ok &= not missing

dupes = collections.Counter(r["target_text"] for r in T["test"])
print(f"duplicate target names within test: "
      f"{sum(1 for v in dupes.values() if v > 1)}")

print("\n" + ("ALL CHECKS PASSED" if ok else "*** PROBLEMS FOUND — do not train ***"))
sys.exit(0 if ok else 1)