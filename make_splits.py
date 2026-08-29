#!/usr/bin/env python3
"""
make_splits.py — build train / val / test splits for fine-tuning.

Two things this protects against:

1. **Eval-set contamination.** The 182 logos you baselined and hand-scored are
   the only items with a human ceiling attached. They are forced into TEST and
   never appear in training.

2. **Name memorization (Tier B).** Training on ~5,000 band names risks teaching
   the model the name list rather than how to read lettering. Test names are
   made token-disjoint from training names, so a model cannot score by recalling
   a name fragment it memorized.

    python make_splits.py --corpus corpus.csv --manifest logos.csv \
        --baseline baseline.csv --out-dir splits

Outputs splits/{train,val,test}.csv (image_path,target_text) plus splits.csv
with every row's assignment and reason.
"""
import argparse, csv, os, random, sys, collections

# Tokens too common to enforce disjointness on — excluding every band whose name
# contains "OF" would gut the training set for no benefit.
STOP = {"THE", "OF", "AND", "A", "IN", "TO", "MY", "IS", "ON", "FOR", "AT",
        "DE", "LA", "EL", "DER", "DIE", "DAS", "NO", "UN"}
MIN_TOKEN = 4          # only enforce on tokens of this length or longer


def norm(s):
    return "".join(c for c in (s or "").upper() if c.isalpha() or c == " ").strip()


def tokens(name):
    return {t for t in norm(name).split()
            if len(t) >= MIN_TOKEN and t not in STOP}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.csv")
    ap.add_argument("--manifest", default="logos.csv")
    ap.add_argument("--baseline", default=None,
                    help="baseline.csv — its images are forced into TEST")
    ap.add_argument("--out-dir", default="splits")
    ap.add_argument("--test-size", type=int, default=400)
    ap.add_argument("--val-size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting an existing splits directory")
    a = ap.parse_args()

    # A training run is only interpretable against the exact splits it used.
    # Refuse to silently replace them.
    existing = [f for f in ("train.csv", "val.csv", "test.csv", "splits.csv")
                if os.path.exists(os.path.join(a.out_dir, f))]
    if existing and not a.force:
        print(f"{a.out_dir}/ already contains: {', '.join(existing)}")
        print("Refusing to overwrite. Either use a new --out-dir "
              "(e.g. --out-dir splits_v2), or pass --force if you are certain.")
        return 1

    rng = random.Random(a.seed)

    # manifest = what actually downloaded
    have = []
    with open(a.manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("image_path") and r.get("target_text"):
                have.append((r["image_path"], r["target_text"]))
    by_path = dict(have)
    print(f"manifest: {len(have)} downloaded logos")

    # corpus metadata, keyed by target text
    meta = {}
    with open(a.corpus, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta[r["target_text"]] = r
    print(f"corpus:   {len(meta)} rows of metadata")

    # protected eval set
    protected = set()
    if a.baseline and os.path.exists(a.baseline):
        with open(a.baseline, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("image_path"):
                    protected.add(r["image_path"])
        protected &= set(by_path)
        print(f"protected: {len(protected)} baselined images forced into TEST")
    else:
        print("protected: none (no --baseline given)")

    rows = [{"image_path": p, "target_text": t,
             "tier": meta.get(t, {}).get("obscurity_tier", ""),
             "genre": meta.get(t, {}).get("primaryGenre", "")}
            for p, t in have]

    # --- assemble TEST: protected first, then obscure-tier fill ---
    test = [r for r in rows if r["image_path"] in protected]
    for r in test:
        r["reason"] = "baselined"
    pool = [r for r in rows if r["image_path"] not in protected]
    rng.shuffle(pool)

    need = max(0, a.test_size - len(test))
    obscure = [r for r in pool if r["tier"] == "obscure"]
    extra = obscure[:need] or pool[:need]
    for r in extra:
        r["reason"] = "test_fill"
    test += extra
    taken = {r["image_path"] for r in test}
    pool = [r for r in pool if r["image_path"] not in taken]

    # --- Tier B: drop training rows sharing a name token with any test name ---
    test_tokens = set()
    for r in test:
        test_tokens |= tokens(r["target_text"])

    keep, dropped = [], 0
    for r in pool:
        if tokens(r["target_text"]) & test_tokens:
            dropped += 1
            continue
        keep.append(r)
    print(f"\ntoken-disjointness: dropped {dropped} rows sharing a name token "
          f"with the test set ({len(test_tokens)} distinct test tokens)")

    rng.shuffle(keep)
    val = keep[: a.val_size]
    train = keep[a.val_size:]
    for r in val:
        r["reason"] = "val"
    for r in train:
        r["reason"] = "train"

    os.makedirs(a.out_dir, exist_ok=True)
    for name, group in (("train", train), ("val", val), ("test", test)):
        with open(os.path.join(a.out_dir, f"{name}.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "target_text"])
            w.writerows([(r["image_path"], r["target_text"]) for r in group])

    with open(os.path.join(a.out_dir, "splits.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "target_text", "tier",
                                          "genre", "split", "reason"])
        w.writeheader()
        for name, group in (("train", train), ("val", val), ("test", test)):
            for r in group:
                w.writerow({**r, "split": name})

    print(f"\ntrain {len(train)}   val {len(val)}   test {len(test)}")
    print(f"  (of test, {len(protected)} are your baselined + hand-scored set)")

    for label, group in (("test", test), ("train", train)):
        c = collections.Counter(r["genre"] for r in group)
        print(f"\n{label} genre mix: " +
              "  ".join(f"{k or '?'}={v}" for k, v in c.most_common(6)))

    with open(os.path.join(a.out_dir, "SPLIT_CONFIG.txt"), "w") as f:
        f.write(f"seed={a.seed}\ntest_size={a.test_size}\nval_size={a.val_size}\n"
                f"corpus={a.corpus}\nmanifest={a.manifest}\nbaseline={a.baseline}\n"
                f"train={len(train)}\nval={len(val)}\ntest={len(test)}\n"
                f"protected={len(protected)}\ntoken_dropped={dropped}\n")

    print(f"\nwrote {a.out_dir}/train.csv, val.csv, test.csv, splits.csv, "
          f"SPLIT_CONFIG.txt")
    print("\nSANITY CHECK before training: confirm no image_path appears in "
          "more than one split, and that test.csv contains your 182.")
    return 0


if __name__ == "__main__":
    sys.exit(main())