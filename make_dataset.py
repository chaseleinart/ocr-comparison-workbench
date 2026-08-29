#!/usr/bin/env python3
"""
make_dataset.py — convert splits/*.csv into JSONL for VLM fine-tuning, and pin
the prompt string so training and evaluation can never drift apart.

    python make_dataset.py --splits-dir splits --out-dir data

Writes data/{train,val,test}.jsonl with one object per line:

    {"image_path": "logos/1234.jpg", "prompt": "...", "completion": "PATHOLOGY"}

and data/PROMPT.txt containing the exact prompt used. Load that same file at
eval time — a prompt that drifts between training and inference looks like a
broken model and is the single most common cause of "my fine-tune got worse".

The JSONL is intentionally schema-neutral. TRL's expected message format varies
by version, so build the final Dataset inside your notebook from these fields —
see the snippet printed at the end.
"""
import argparse, csv, json, os, statistics, sys

PROMPT = "Transcribe the band name written in this logo. Reply with only the text."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    existing = [f for f in ("train.jsonl", "val.jsonl", "test.jsonl")
                if os.path.exists(os.path.join(a.out_dir, f))]
    if existing and not a.force:
        print(f"{a.out_dir}/ already contains: {', '.join(existing)}")
        print("Use a new --out-dir or pass --force.")
        return 1

    os.makedirs(a.out_dir, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        Image = None

    totals = {}
    for split in ("train", "val", "test"):
        src = os.path.join(a.splits_dir, f"{split}.csv")
        rows = list(csv.DictReader(open(src, newline="", encoding="utf-8")))
        out = os.path.join(a.out_dir, f"{split}.jsonl")
        widths, heights, bad = [], [], 0
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                p, t = r["image_path"], r["target_text"].strip().upper()
                if Image:
                    try:
                        with Image.open(p) as im:
                            widths.append(im.width)
                            heights.append(im.height)
                    except Exception:
                        bad += 1
                        continue
                f.write(json.dumps({"image_path": p, "prompt": a.prompt,
                                    "completion": t}) + "\n")
        totals[split] = len(rows) - bad
        line = f"{split:<6}{totals[split]:>6} examples"
        if bad:
            line += f"   ({bad} unreadable, skipped)"
        if widths:
            line += (f"   width med {int(statistics.median(widths))} "
                     f"[{min(widths)}-{max(widths)}]  "
                     f"height med {int(statistics.median(heights))}")
        print(line)

    with open(os.path.join(a.out_dir, "PROMPT.txt"), "w", encoding="utf-8") as f:
        f.write(a.prompt)

    print(f"\nwrote {a.out_dir}/train.jsonl, val.jsonl, test.jsonl, PROMPT.txt")
    print("\nIn the notebook, build the Dataset from these fields:\n")
    print('''    from datasets import load_dataset
    from PIL import Image

    PROMPT = open("data/PROMPT.txt").read()

    def to_messages(ex):
        return {
            "images": [Image.open(ex["image_path"]).convert("RGB")],
            "messages": [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": ex["prompt"]}]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": ex["completion"]}]},
            ],
        }

    ds = load_dataset("json", data_files={
        "train": "data/train.jsonl", "validation": "data/val.jsonl"})
    ds = ds.map(to_messages, remove_columns=ds["train"].column_names)''')
    print("\nCheck that message shape against your TRL version's example "
          "notebook — the schema has changed between releases.")
    return 0


if __name__ == "__main__":
    sys.exit(main())