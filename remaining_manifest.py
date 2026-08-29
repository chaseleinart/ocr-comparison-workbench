#!/usr/bin/env python3
"""
remaining_manifest.py — emit the manifest rows you have NOT already reviewed.

The human-ceiling app shuffles its queue, so a completed session is a random
subset, not a prefix. This diffs by target text (or image_path if your export
has it) and writes what's left.

    python remaining_manifest.py --manifest logos.csv --seen seen_targets.txt \
        --out logos_remaining.csv --corpus corpus.csv

--seen accepts either:
  * a plain text file, one target per line, or
  * the CSV exported by the app (uses image_path if present, else target)

--corpus is optional; if given, obscurity_tier is joined in so the app can show
it and you can stratify later.
"""
import argparse, csv, os, sys


def norm(s):
    return "".join(c for c in (s or "").upper() if c.isalpha())


def load_seen(path):
    seen_paths, seen_targets = set(), set()
    with open(path, newline="", encoding="utf-8") as f:
        head = f.readline()
        f.seek(0)
        if "," in head and ("target" in head.lower() or "image_path" in head.lower()):
            for r in csv.DictReader(f):
                if r.get("image_path"):
                    seen_paths.add(os.path.basename(r["image_path"]))
                if r.get("target"):
                    seen_targets.add(norm(r["target"]))
                elif r.get("target_text"):
                    seen_targets.add(norm(r["target_text"]))
        else:
            for line in f:
                if line.strip():
                    seen_targets.add(norm(line))
    return seen_paths, seen_targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="logos.csv")
    ap.add_argument("--seen", required=True)
    ap.add_argument("--out", default="logos_remaining.csv")
    ap.add_argument("--corpus", default=None)
    a = ap.parse_args()

    seen_paths, seen_targets = load_seen(a.seen)
    print(f"seen: {len(seen_paths)} paths, {len(seen_targets)} distinct targets")

    tiers = {}
    if a.corpus and os.path.exists(a.corpus):
        with open(a.corpus, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                tiers[norm(r.get("target_text"))] = r.get("obscurity_tier", "")

    rows, skipped = [], 0
    with open(a.manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p, t = r.get("image_path", ""), r.get("target_text", "")
            if os.path.basename(p) in seen_paths or norm(t) in seen_targets:
                skipped += 1
                continue
            row = {"image_path": p, "target_text": t}
            if tiers:
                row["obscurity_tier"] = tiers.get(norm(t), "")
            rows.append(row)

    if not rows:
        print("Nothing left to review.")
        return 0

    fields = list(rows[0].keys())
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"skipped {skipped} already-reviewed, wrote {len(rows)} to {a.out}")
    print("\nLoad that manifest in the human-ceiling app along with the same "
          "image folder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())