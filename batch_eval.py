#!/usr/bin/env python3
"""
batch_eval.py — Phase 0 baseline: run every OCR model over a set of logos and
record CER + exact match.

Drop this next to app.py in the workbench and run:

    python batch_eval.py --manifest logos.csv --out baseline.csv

manifest.csv is two columns with a header:

    image_path,target_text
    logos/pathology.jpg,PATHOLOGY
    logos/pharmacist.png,PHARMACIST

Design notes:
  - Resumable. Every result is appended immediately and completed
    (model, image) pairs are skipped on restart. A crash 180 images in
    costs you nothing.
  - Failures are recorded as errors, NOT as empty predictions. An OOM is
    missing data; scoring it as a wrong answer silently corrupts the mean.
  - Models run one at a time with teardown between, since four VLMs will not
    co-reside on most GPUs.
  - DeepSeek's loader is cached (see patch below) because model_service
    reloads it on every single call.
"""
import argparse, asyncio, csv, gc, os, sys, time
from functools import lru_cache

# --- cache DeepSeek's loader -------------------------------------------------
# model_service.run_ocr_inference calls load_model() per image. With Unsloth's
# unsloth_force_compile=True that pays a full load + compile every time. We
# patch the name in model_service's namespace so the registry dispatch is
# otherwise untouched.
def _patch_deepseek_cache():
    try:
        import model_service
    except ImportError:
        return False
    original = model_service.load_model

    @lru_cache(maxsize=1)
    def cached_load(model_name="./deepseek_ocr"):
        print(f"  [loading DeepSeek from {model_name} — once per run]", flush=True)
        return original(model_name=model_name)

    model_service.load_model = cached_load
    return True


# --- scoring -----------------------------------------------------------------

def normalize(s, keep_spaces=False, keep_digits=False):
    """Fold predictions and targets to a comparable form."""
    if s is None:
        return ""
    s = str(s).upper()
    ok = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if keep_spaces:
        ok.add(" ")
    if keep_digits:
        ok |= set("0123456789")
    out = "".join(c for c in s if c in ok)
    return " ".join(out.split()) if keep_spaces else out


def levenshtein(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred, truth):
    """Character error rate. 0.0 is perfect; can exceed 1.0 when the prediction
    is much longer than the target — which is exactly what happens when a model
    returns a paragraph of image description instead of a transcription."""
    if not truth:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, truth) / len(truth)


# --- manifest / resume -------------------------------------------------------

def load_manifest(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = (r.get("image_path") or "").strip()
            t = (r.get("target_text") or "").strip()
            if p and t:
                rows.append((p, t))
    return rows


FIELDS = ["model", "image_path", "target", "pred_raw", "pred_norm",
          "cer", "exact", "status", "seconds"]


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            done.add((r["model"], r["image_path"]))
    return done


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def teardown():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# --- runner ------------------------------------------------------------------

async def run(args):
    if args.dry_run:
        model_names = ["FakeModel"]

        async def infer(model, path):
            await asyncio.sleep(0.01)
            base = os.path.basename(path).rsplit(".", 1)[0]
            if "boom" in base:
                raise RuntimeError("simulated CUDA OOM")
            return base.upper()[:-1]          # drop a char so CER != 0
    else:
        if not _patch_deepseek_cache():
            print("Could not import model_service — run this from the "
                  "workbench directory.", file=sys.stderr)
            return 1
        from model_service import get_all_ocr_model_names, run_ocr_inference
        model_names = get_all_ocr_model_names()

        async def infer(model, path):
            return await run_ocr_inference(model, path)

    if args.models:
        want = {m.strip().lower() for m in args.models.split(",")}
        model_names = [m for m in model_names if m.lower() in want]
        if not model_names:
            print("No models matched --models", file=sys.stderr)
            return 1

    items = load_manifest(args.manifest)
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("Manifest is empty", file=sys.stderr)
        return 1

    done = load_done(args.out)
    print(f"{len(model_names)} model(s) x {len(items)} image(s); "
          f"{len(done)} already done in {args.out}\n")

    for model in model_names:
        pending = [(p, t) for p, t in items if (model, p) not in done]
        if not pending:
            print(f"== {model}: already complete, skipping")
            continue
        print(f"== {model}: {len(pending)} to run")
        for i, (path, truth) in enumerate(pending, 1):
            tnorm = normalize(truth, args.keep_spaces, args.keep_digits)
            t0 = time.time()
            if not os.path.exists(path) and not args.dry_run:
                append_row(args.out, {
                    "model": model, "image_path": path, "target": tnorm,
                    "pred_raw": "", "pred_norm": "", "cer": "", "exact": "",
                    "status": "missing_file", "seconds": 0})
                print(f"  [{i}/{len(pending)}] {os.path.basename(path)}: FILE MISSING")
                continue
            try:
                raw = await infer(model, path)
                pnorm = normalize(raw, args.keep_spaces, args.keep_digits)
                row = {"model": model, "image_path": path, "target": tnorm,
                       "pred_raw": (raw or "")[: args.max_raw].replace("\n", " "),
                       "pred_norm": pnorm,
                       "cer": f"{cer(pnorm, tnorm):.4f}",
                       "exact": int(pnorm == tnorm),
                       "status": "ok", "seconds": f"{time.time()-t0:.1f}"}
                print(f"  [{i}/{len(pending)}] {os.path.basename(path)}: "
                      f"CER {row['cer']}  {pnorm[:28] or '(empty)'}")
            except Exception as e:
                row = {"model": model, "image_path": path, "target": tnorm,
                       "pred_raw": "", "pred_norm": "", "cer": "", "exact": "",
                       "status": f"error: {type(e).__name__}: {e}"[:200],
                       "seconds": f"{time.time()-t0:.1f}"}
                print(f"  [{i}/{len(pending)}] {os.path.basename(path)}: ERROR {e}")
            append_row(args.out, row)
        teardown()
        print()

    summarize(args.out)
    return 0


def summarize(path):
    by = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = by.setdefault(r["model"], {"cers": [], "exact": 0, "err": 0})
            if r["status"] == "ok":
                d["cers"].append(float(r["cer"]))
                d["exact"] += int(r["exact"])
            else:
                d["err"] += 1

    print("=" * 72)
    print(f"{'model':<20}{'n':<6}{'mean CER':<11}{'median':<10}{'exact':<8}errors")
    print("-" * 72)
    for m, d in sorted(by.items(), key=lambda kv: _mean(kv[1]['cers'])):
        c = sorted(d["cers"])
        n = len(c)
        med = c[n // 2] if n else float("nan")
        print(f"{m:<20}{n:<6}{_mean(c):<11.4f}{med:<10.4f}"
              f"{d['exact']:<8}{d['err']}")
    print("=" * 72)
    print("CER: 0.0 perfect, 1.0 = every character wrong. Values >1.0 mean the "
          "model returned far more text than the target (image descriptions).")
    print("Errors are excluded from the means, not scored as failures.")


def _mean(a):
    return sum(a) / len(a) if a else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="logos.csv",
                    help="CSV with image_path,target_text")
    ap.add_argument("--out", default="baseline.csv")
    ap.add_argument("--models", default=None,
                    help="comma-separated subset, e.g. 'Chandra OCR,Qwen3-VL'")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep-spaces", action="store_true")
    ap.add_argument("--keep-digits", action="store_true")
    ap.add_argument("--max-raw", type=int, default=300,
                    help="truncate stored raw output; descriptions get long")
    ap.add_argument("--dry-run", action="store_true",
                    help="fake backend, exercises the harness with no GPU")
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()

    if a.summarize_only:
        summarize(a.out)
        return 0
    return asyncio.run(run(a))


if __name__ == "__main__":
    sys.exit(main())