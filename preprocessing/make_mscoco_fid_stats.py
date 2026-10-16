"""Compute the MS-COCO 256 val FID statistics and compare them with U-ViT's fid_stats_mscoco256_val.npz.

U-ViT's stats: https://huggingface.co/datasets/SaeedRazavi/mscoco256-t2i-features (fid_stats/)

Usage:
    uv run python preprocessing/make_mscoco_fid_stats.py --features data/mscoco256/features/val \
        --out data/mscoco256/fid_stats/fid_stats_mscoco256_val_recomputed.npz \
        --official data/mscoco256/fid_stats/fid_stats_mscoco256_val.npz
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True,
                        help="arrow val split (images are the center-cropped 256x256 pixels)")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--official", type=str, default=None,
                        help="U-ViT's fid_stats_mscoco256_val.npz, to compare against")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    import torch
    from datasets import load_from_disk
    from recon.eval.t2i import compute_inception_stats, calculate_frechet_distance

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_from_disk(args.features)
    print(f"{len(ds)} val images from {args.features}")

    images = np.stack([
        np.asarray(row["image"].convert("RGB"), dtype=np.uint8)
        for row in tqdm(ds, desc="loading val images")
    ])
    mu, sigma = compute_inception_stats(images, device=device,
                                        batch_size=args.batch_size, verbose=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, mu=mu, sigma=sigma)
    print(f"saved {args.out}")

    if args.official is not None:
        ref = np.load(args.official)
        fid = calculate_frechet_distance(mu, sigma, ref["mu"], ref["sigma"])
        print(f"FID(recomputed, official) = {fid:.4f}  (should be ~0; <0.5 is a pass)")


if __name__ == "__main__":
    main()
