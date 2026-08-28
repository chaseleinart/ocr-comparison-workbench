#!/usr/bin/env python3
"""
prepare_corpus.py — turn the MetalVis console dump into a downloaded image set
plus a manifest that batch_eval.py can consume.

    # 1. parse only, no network — see what survives cleaning
    python prepare_corpus.py --dump log-of_band-logo-data.txt --parse-only

    # 2. download a stratified 200-logo baseline set
    python prepare_corpus.py --dump log-of_band-logo-data.txt \
        --out-dir logos --sample 200

    # 3. later: download everything for training
    python prepare_corpus.py --dump log-of_band-logo-data.txt --out-dir logos

Outputs:
    corpus.csv   full metadata for every parsed band (one row per unique id)
    logos.csv    image_path,target_text — the manifest for batch_eval.py

Scraping etiquette: requests are serialized with a delay (default 1.5s) and
every file is cached on disk, so a re-run downloads nothing. Do not lower the
delay. Metal Archives is a volunteer-run archive.
"""
import argparse, csv, hashlib, os, random, re, sys, time, unicodedata
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# NOTE: this export carries 15 columns. The trailing two are grid coordinates
# (gx, gy) from the MetalVis scatterplot — NOT image dimensions. The logo /
# width / height fields present in the full MetalVis schema are absent here,
# so real image size is only knowable after download.
COLS = ["index", "id", "name", "formed", "genre", "country", "active",
        "themes", "label", "url", "x", "y", "primaryGenre", "gx", "gy"]

UA = ("Mozilla/5.0 (compatible; logo-ocr-research/0.1; "
      "personal research; contact via github)")


# ------------------------------- parsing ------------------------------------

def parse_dump(path):
    """Extract tab-delimited data rows from the browser console log, ignoring
    the surrounding noise (WebGL warnings, source-map errors, object reprs)."""
    rows, seen = [], set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not re.match(r"^\d+\t", line):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 15:
                continue
            r = dict(zip(COLS, parts[:15]))
            # the dump pages in blocks of 1000 so `index` repeats; id is the key
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            rows.append(r)
    return rows


# --------------------------- target text rules -------------------------------

LATIN_OK = re.compile(r"^[A-Za-z0-9 &'\-\.!/\+\(\)]+$")


def fold(s):
    """Strip diacritics: Mötley -> Motley. Logos usually render the umlaut, but
    we normalize both sides at scoring time, so this only affects the target."""
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def make_target(name, keep_spaces=True):
    t = fold(name).upper()
    ok = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ" + (" " if keep_spaces else ""))
    t = "".join(c if c in ok else (" " if c in "-_/&." else "") for c in t)
    return " ".join(t.split())


def classify(r):
    """Return (ok, reason). Rejections are counted, not silently dropped."""
    name = r["name"].strip()
    if not name:
        return False, "empty_name"
    if not LATIN_OK.match(fold(name)):
        return False, "non_latin_script"
    tgt = make_target(name)
    if len(tgt.replace(" ", "")) < 3:
        return False, "target_too_short"
    if len(tgt) > 40:
        return False, "target_too_long"
    if not r["url"].startswith("http"):
        return False, "no_url"
    return True, ""


# ------------------------------ downloading ----------------------------------

def fetch(url, dest, delay, retries=3):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "cached"
    for a in range(retries):
        try:
            req = Request(url, headers={"User-Agent": UA})
            with urlopen(req, timeout=30) as r:
                data = r.read()
            if len(data) < 200:
                return "too_small"
            with open(dest, "wb") as f:
                f.write(data)
            time.sleep(delay)
            return "downloaded"
        except HTTPError as e:
            if e.code == 404:
                return "http_404"
            time.sleep(delay * (a + 2))
        except (URLError, OSError) as e:
            time.sleep(delay * (a + 2))
    return "failed"


def image_dims(path):
    """Real dimensions, read from the downloaded file. Returns None if the
    bytes aren't a decodable image."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------- main --------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out-dir", default="logos")
    ap.add_argument("--corpus-csv", default="corpus.csv")
    ap.add_argument("--manifest", default="logos.csv")
    ap.add_argument("--sample", type=int, default=None,
                    help="download only N, stratified across the obscurity range")
    ap.add_argument("--delay", type=float, default=1.5,
                    help="seconds between requests — do not lower")
    ap.add_argument("--parse-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-width", type=int, default=80)
    a = ap.parse_args()

    rows = parse_dump(a.dump)
    print(f"parsed {len(rows)} unique bands")

    kept, rejects = [], {}
    for r in rows:
        ok, why = classify(r)
        if ok:
            r["target_text"] = make_target(r["name"])
            kept.append(r)
        else:
            rejects[why] = rejects.get(why, 0) + 1

    print("\nrejected:")
    for k, v in sorted(rejects.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<20}{v}")
    print(f"\n{len(kept)} usable rows\n")

    # Obscurity proxy: Metal Archives ids are assigned in registration order, so
    # low ids skew famous (Amorphis=1, Blind Guardian=3). This is CRUDE — a
    # 2003-registered obscure band gets a low id too. Replace it with a real
    # popularity signal before trusting any train/test split built on it.
    for r in kept:
        r["ma_id"] = int(r["id"])
    kept.sort(key=lambda r: r["ma_id"])
    n = len(kept)
    for i, r in enumerate(kept):
        r["obscurity_rank"] = i
        r["obscurity_tier"] = ("famous_ish" if i < n * 0.25
                               else "mid" if i < n * 0.75 else "obscure")

    with open(a.corpus_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS + ["target_text", "ma_id",
                                                 "obscurity_rank", "obscurity_tier"])
        w.writeheader()
        for r in kept:
            w.writerow(r)
    print(f"wrote {a.corpus_csv}")

    if a.parse_only:
        print("\n--parse-only: stopping before any network access.")
        print("Sample targets:")
        for r in random.Random(a.seed).sample(kept, min(12, len(kept))):
            print(f"  {r['target_text']:<28} <- {r['name']}")
        return 0

    todo = kept
    if a.sample:
        # stratify so the sample spans the obscurity range rather than
        # clustering in whichever tier happens to sort first
        rnd = random.Random(a.seed)
        per = {}
        for r in kept:
            per.setdefault(r["obscurity_tier"], []).append(r)
        todo, share = [], max(1, a.sample // len(per))
        for tier, group in per.items():
            todo += rnd.sample(group, min(share, len(group)))
        rnd.shuffle(todo)
        todo = todo[: a.sample]

    os.makedirs(a.out_dir, exist_ok=True)
    print(f"\ndownloading {len(todo)} logos to {a.out_dir}/ "
          f"({a.delay}s between requests, ~{len(todo)*a.delay/60:.0f} min)\n")

    manifest, hashes, stats = [], {}, {}
    for i, r in enumerate(todo, 1):
        ext = os.path.splitext(r["url"].split("?")[0])[1].lower() or ".jpg"
        if ext not in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
            ext = ".jpg"
        dest = os.path.join(a.out_dir, f"{r['id']}{ext}")
        status = fetch(r["url"], dest, a.delay)
        stats[status] = stats.get(status, 0) + 1

        if status in ("downloaded", "cached"):
            dims = image_dims(dest)
            if dims is None:
                stats["undecodable"] = stats.get("undecodable", 0) + 1
                os.remove(dest)
                continue
            w, h_px = dims
            if w < 80 or h_px < 25:
                stats["too_small"] = stats.get("too_small", 0) + 1
                os.remove(dest)
                continue
            h = file_hash(dest)
            if h in hashes:
                stats["duplicate_image"] = stats.get("duplicate_image", 0) + 1
                os.remove(dest)
                status = "duplicate"
            else:
                hashes[h] = r["id"]
                manifest.append((dest, r["target_text"]))
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  " +
                  "  ".join(f"{k}={v}" for k, v in sorted(stats.items())),
                  flush=True)

    with open(a.manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "target_text"])
        w.writerows(manifest)

    print(f"\nwrote {a.manifest} with {len(manifest)} rows")
    print("Next: python batch_eval.py --manifest "
          f"{a.manifest} --out baseline.csv --dry-run")
    print("\nBefore training on this, eyeball 100 rows yourself — the band name "
          "is not always what the logo actually renders.")
    return 0


if __name__ == "__main__":
    sys.exit(main())