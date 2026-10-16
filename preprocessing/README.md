# Preprocessing Guide

All datasets and FID references are available preprocessed on the Hugging Face Hub. Download them into
`data/` with `uv run hf download`, or build them yourself with the scripts in this folder. Commands are run
from the repository root and produce the `data/` layout shown in the [main README](../README.md).

## ImageNet 256×256 (class-to-image)

**Images and latents.** `c2i_train.py` reads two datasets saved with `save_to_disk` under `--data-dir`
(`data/imagenet`): the images with their labels, and the VAE latents of the same images. The
[G-REPA](https://huggingface.co/G-REPA) organization hosts both in this format:

```bash
uv run hf download G-REPA/imagenet-latents-images --repo-type dataset --local-dir data/imagenet/imagenet-latents-images
uv run hf download G-REPA/imagenet-latents-sdvae-ft-mse-f8d4 --repo-type dataset --local-dir data/imagenet/imagenet-latents-sdvae-ft-mse-f8d4
```

Latents for other VAEs (`--vae-name`) go to `data/imagenet/imagenet-latents-<vae-name>`:

| `--vae-name` | Hugging Face dataset |
|---|---|
| `sdvae-ft-mse-f8d4` | `G-REPA/imagenet-latents-sdvae-ft-mse-f8d4` |
| `sdvae-ft-ema-f8d4` | `G-REPA/imagenet-latents-sdvae-ft-mse-f8d4` (same encoder as `sdvae-ft-mse`) |
| `e2e-sdvae-f8d4` | `G-REPA/imagenet-latents-davae-sdvae-mse-align1.5-400k` |
| `invae-f16d32` | `G-REPA/imagenet-latents-invae-f16d32` |
| `vavae-f16d32` | `G-REPA/imagenet-latents-vavae-f16d32` |
| `e2e-vavae-f16d32` | `G-REPA/imagenet-latents-davae-vavae-align1.5-400k` |

**FID reference.** ImageNet results are reported against ADM's reference batch
([guided-diffusion](https://github.com/openai/guided-diffusion/tree/main/evaluations)); point the
`REFERENCE_NPZ` environment variable at it:

```bash
wget -P data/imagenet https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
```

## MS-COCO 256×256 (text-to-image, MMDiT)

**Features.** U-ViT's preprocessing of MS-COCO 2014: 256×256 center crops, SD-VAE posteriors and CLIP
ViT-L/14 caption embeddings, plus the embedding of the empty caption. Download them with the FID
reference:

```bash
uv run hf download SaeedRazavi/mscoco256-t2i-features --repo-type dataset --local-dir data/mscoco256
```

This gives `data/mscoco256/features/` (`--data-dir`) and `data/mscoco256/fid_stats/fid_stats_mscoco256_val.npz`
(`--ref-stats`). Or extract the features yourself from the COCO 2014 release, placed in `data/coco/`
(`train2014/`, `val2014/`, `annotations/`):

```bash
uv run python preprocessing/extract_mscoco_features.py --split train --coco-root data/coco --out data/mscoco256/features
uv run python preprocessing/extract_mscoco_features.py --split val   --coco-root data/coco --out data/mscoco256/features
```

These are SD-VAE latents. For another VAE, add `--vae-name <name>` with a new `--out` directory and train with
the same `--vae-name`.

**FID reference.** U-ViT's Inception statistics of the val images (downloaded above), or compute them
yourself:

```bash
uv run python preprocessing/make_mscoco_fid_stats.py --features data/mscoco256/features/val \
    --out data/mscoco256/fid_stats/fid_stats_mscoco256_val.npz
```

`--official <U-ViT's npz>` also reports the FID between the two (0.00 for ours).

## BLIP3o 256×256 (text-to-image, DDT)

**Shards.** The BLIP3o pretraining data at 256×256 as WebDataset shards, one directory per split
(`journeydb`, `long-caption`, `short-caption`; 1.3 TB):

```bash
uv run hf download diffusion-bench/blip3o-256 --repo-type dataset --local-dir data/blip3o-256 \
    --include "journeydb/*" --include "long-caption/*" --include "short-caption/*"
```

**FID reference.** DiffusionBench's BLIP3o FID-50k protocol: 50,000 (image, caption) pairs drawn from the
training shards in proportion to the splits. Their images give the Inception statistics
(`reference_stats.npz`, `--ref-stats`) and their captions are the prompts (`prompts/val`,
`--eval-prompts`). Download the frozen reference (55 MB):

```bash
uv run hf download SaeedRazavi/blip3o-256-fid50k --repo-type dataset --local-dir data/blip3o-256-fid50k \
    --include "reference_stats.npz" --include "prompts/*"
```

or build it yourself from the shards (the defaults reproduce it):

```bash
uv run python preprocessing/build_blip3o_reference.py --data-dir data/blip3o-256 --out-dir data/blip3o-256-fid50k
```
