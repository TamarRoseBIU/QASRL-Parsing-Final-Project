#!/usr/bin/env python3
"""
Download the base QA-SRL data splits used by SFT and GRPO.

The training scripts (training/sft/Stage_CE_Instruct_DEV.py,
training/grpo/Stage_GRPO_Instruct_DEV.py) fetch these same splits from the URL below
at runtime; this script simply materializes them locally so the data is available
offline and its provenance is explicit.

Splits (V-passive_red):
  train.json  -- full 92k noisy split (used only by the SFT TRAIN pipeline)
  dev.json    -- clean split; the SFT/GRPO models are trained on THIS
  test.json   -- held-out evaluation split

Usage:
  python download_data.py                 # -> ./raw/{train,dev,test}.json
  python download_data.py --out-dir DIR    # custom destination
  python download_data.py --splits dev test
"""
import argparse
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://nlp.biu.ac.il/~ron.eliav/qasrl/V-passive_red"
SPLITS = ("train", "dev", "test")


def download(split: str, out_dir: Path) -> Path:
    url = f"{BASE_URL}/{split}.json"
    dest = out_dir / f"{split}.json"
    print(f"[download] {url} -> {dest}")
    with urllib.request.urlopen(url) as resp:  # nosec - fixed, trusted host
        data = resp.read()
    dest.write_bytes(data)
    print(f"[download]   wrote {len(data):,} bytes")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "raw",
                    help="destination directory (default: ./raw next to this script)")
    ap.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS),
                    help="which splits to download (default: all)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        download(split, args.out_dir)
    print(f"[download] done -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
