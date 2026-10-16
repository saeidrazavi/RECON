"""Build the frozen FID reference of the DDT eval (the BLIP3o FID-50k protocol of DiffusionBench).

Samples --num-ref (image, caption) pairs from the BLIP3o-256 training shards and writes to --out-dir:

    reference_stats.npz   mu / sigma of the torch-fidelity Inception features
    prompts/val/          the captions of the samples, used as generation prompts
    manifest.json         seed, quotas, shards and the key of every sample

Prebuilt reference: https://huggingface.co/datasets/SaeedRazavi/blip3o-256-fid50k
To rebuild it, run the script with the default arguments:

    uv run python preprocessing/build_blip3o_reference.py --data-dir data/blip3o-256 --out-dir data/blip3o-256-fid50k
"""
import argparse
import io
import json
import os
import random
import sys
import tarfile
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recon.eval.c2i import _inception_features

# training-set sizes of the splits, which set the per-split quotas
SPLITS = {
    "journeydb": 4_280_000,
    "long-caption": 27_200_000,
    "short-caption": 4_770_000,
}
IMAGE_SIZE = 256


def proportional_quotas(num_ref):
    """Per-split quotas proportional to the training-set sizes, summing to num_ref."""
    total = sum(SPLITS.values())
    quotas = {s: int(num_ref * n / total) for s, n in SPLITS.items()}
    quotas[max(SPLITS, key=SPLITS.get)] += num_ref - sum(quotas.values())
    return quotas


def reservoir_sample_shard(shard_path, k, rng):
    """Uniform sample of k (key, caption, image bytes) records from one tar shard, in a single pass."""
    reservoir, seen = [], 0
    pending = {}  # stem -> partial {img, txt}
    with tarfile.open(shard_path) as tf:
        for member in tf:
            if not member.isfile():
                continue
            stem, ext = os.path.splitext(member.name)
            ext = ext.lower()
            if ext in (".jpg", ".jpeg", ".png", ".webp"):
                pending.setdefault(stem, {})["img"] = tf.extractfile(member).read()
            elif ext == ".txt":
                pending.setdefault(stem, {})["txt"] = tf.extractfile(member).read().decode("utf-8").strip()
            else:
                continue
            entry = pending.get(stem)
            if entry and "img" in entry and "txt" in entry:
                pending.pop(stem)
                if not entry["txt"]:
                    continue
                sample = {"key": stem, "caption": entry["txt"], "img": entry["img"]}
                if len(reservoir) < k:
                    reservoir.append(sample)
                else:
                    j = rng.randint(0, seen)  # inclusive
                    if j < k:
                        reservoir[j] = sample
                seen += 1
    return reservoir


def decode_to_256(image_bytes):
    """256x256 RGB uint8; resize the shorter side and center-crop only if needed."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        w, h = img.size
        scale = IMAGE_SIZE / min(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
        w, h = img.size
        left, top = (w - IMAGE_SIZE) // 2, (h - IMAGE_SIZE) // 2
        img = img.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
    return np.asarray(img, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=str, required=True, help="BLIP3o-256 root with one directory per split")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--num-ref", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--samples-per-shard", type=int, default=500,
                        help="Target reservoir size per shard; sets how many shards are read")
    parser.add_argument("--device", type=str, default="cuda", help="Device of the Inception pass")
    parser.add_argument("--stats-batch-size", type=int, default=128)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    stats_path = out_dir / "reference_stats.npz"
    prompts_path = out_dir / "prompts" / "val"
    manifest_path = out_dir / "manifest.json"
    if stats_path.exists() or (prompts_path / "dataset_info.json").exists():
        sys.exit(f"Refusing to overwrite the reference in {out_dir}: FID values computed against it "
                 "would no longer be comparable. Delete it yourself to rebuild.")
    out_dir.mkdir(parents=True, exist_ok=True)

    quotas = proportional_quotas(args.num_ref)
    print(f"Total {args.num_ref} reference samples, quotas: {quotas}")
    records = []
    manifest_shards = {}
    for split, quota in quotas.items():
        shards = sorted(glob(os.path.join(args.data_dir, split, "*.tar")))
        assert shards, f"no shards for {split}"
        n_shards = max(2, min(len(shards), -(-quota // args.samples_per_shard)))  # ceil
        per_shard = -(-quota // n_shards)
        rng = random.Random(f"{args.seed}-{split}")
        chosen = rng.sample(shards, n_shards)
        manifest_shards[split] = [os.path.basename(s) for s in chosen]
        print(f"[{split}] quota {quota}: {per_shard}/shard from {n_shards} of {len(shards)} shards")

        split_records = []
        for i, shard in enumerate(chosen):
            shard_rng = random.Random(f"{args.seed}-{split}-{os.path.basename(shard)}")
            got = reservoir_sample_shard(shard, per_shard, shard_rng)
            for s in got:
                s["split"], s["shard"] = split, os.path.basename(shard)
            split_records.extend(got)
            print(f"  [{split} {i + 1}/{n_shards}] {os.path.basename(shard)}: +{len(got)} "
                  f"({len(split_records)}/{quota})", flush=True)
        assert len(split_records) >= quota, f"{split}: sampled {len(split_records)} < quota {quota}"
        rng.shuffle(split_records)
        records.extend(split_records[:quota])
    random.Random(args.seed).shuffle(records)

    print(f"Sampled {len(records)} pairs; decoding images...")
    images = np.stack([decode_to_256(r["img"]) for r in records])  # (N, 256, 256, 3) uint8
    for r in records:
        del r["img"]
    print(f"Computing Inception statistics of {images.shape} on {args.device}...")
    feats, _ = _inception_features(images, args.stats_batch_size, args.device)
    np.savez(stats_path, mu=feats.mean(axis=0), sigma=np.cov(feats, rowvar=False))
    print(f"Wrote {stats_path}")

    from datasets import Dataset
    Dataset.from_dict({"text": [r["caption"] for r in records]}).save_to_disk(str(prompts_path))
    print(f"Wrote the prompts ({len(records)} rows) to {prompts_path}")
    with open(manifest_path, "w") as f:
        json.dump({
            "num_ref": len(records),
            "seed": args.seed,
            "samples_per_shard": args.samples_per_shard,
            "quotas": quotas,
            "shards": manifest_shards,
            "rows": [{"split": r["split"], "shard": r["shard"], "key": r["key"]} for r in records],
        }, f)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
