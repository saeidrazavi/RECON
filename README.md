<h1 align="center">RECON: Representation Conditioning for Generation</h1>

<p align="center">
  <a href="https://scholar.google.com/citations?user=5I-A3XsAAAAJ&hl=en" target="_blank">Saeed&nbsp;Razavi</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=GQzvqS4AAAAJ&hl=en" target="_blank">Xingjian&nbsp;Leng</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=vNHqr3oAAAAJ&hl=en" target="_blank">Liang&nbsp;Zheng</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Australian National University
</p>

<p align="center">
  <a href="https://recondiffusion.github.io/">🌐 Project Page</a> &ensp;
  <a href="https://huggingface.co/SaeedRazavi/RECON">🤗 Models</a> &ensp;
  <a href="https://arxiv.org/abs/2610.20608">📃 Paper</a>
</p>

<p align="center">
  <img src="assets/recon.gif" alt="RECON: Representation Conditioning for Generation" width="720">
</p>

This repository provides the official implementation of RECON (Representation Conditioning).
[REPA](https://github.com/sihyun-yu/REPA) trains an extra MLP head to align a diffusion transformer with a frozen
vision encoder (the *teacher*), and then discards the head for generation. RECON reuses the prediction of this
head as a condition for generation. The codebase trains and evaluates three settings:

| Task | Model | Data | Evaluation |
|---|---|---|---|
| Class-to-image | SiT | ImageNet 256×256 | FID / IS / precision / recall (ADM) |
| Text-to-image | MMDiT | MS-COCO 256×256 | FID (U-ViT) + CLIP score |
| Text-to-image | DDT | BLIP3o 256×256 | FID (DiffusionBench) + CLIP score |

Please find further details in our [blog post](https://recondiffusion.github.io/).

## 🛠️ 1. Environment setup

```bash
git clone https://github.com/saeidrazavi/RECON.git
cd RECON

# install uv project manager (if you do not have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# install dependencies
uv sync
```

All commands run from the repository root, laid out as follows once the data is in place:

```
RECON/
├── c2i_train.py, c2i_eval.py        # class-to-image: SiT on ImageNet
├── t2i_train.py, t2i_eval.py        # text-to-image: MMDiT on MS-COCO, DDT on BLIP3o
├── recon/
│   ├── models/                      # SiT, MMDiT, DDT; VAEs; Qwen3 text encoder
│   ├── encoders/                    # teacher encoders (--enc-type)
│   ├── data/                        # ImageNet, MS-COCO and BLIP3o loaders
│   ├── diffusion/                   # loss and samplers
│   └── eval/                        # FID / IS / precision / recall, CLIP score
├── preprocessing/                   # scripts that rebuild the data and FID references
├── data/
│   ├── imagenet/                    # --data-dir (class-to-image)
│   │   ├── imagenet-latents-images/
│   │   ├── imagenet-latents-sdvae-ft-mse-f8d4/       # one directory per --vae-name
│   │   └── VIRTUAL_imagenet256_labeled.npz           # REFERENCE_NPZ
│   ├── mscoco256/
│   │   ├── features/                # --data-dir (MMDiT): train/, val/, empty_context.npy
│   │   └── fid_stats/fid_stats_mscoco256_val.npz     # --ref-stats
│   ├── blip3o-256/                  # --data-dir (DDT): journeydb/, long-caption/, short-caption/
│   └── blip3o-256-fid50k/           # reference_stats.npz (--ref-stats), prompts/val/ (--eval-prompts)
├── pretrained_models/               # optional teacher weights (MoCo v3, MAE, I-JEPA, DINOv3)
├── dinov3/                          # optional clone of the DINOv3 repository
└── exps/                            # checkpoints, logs and evals per --exp-name
```

## 📦 2. Data

All datasets and FID references are on the Hugging Face Hub. The
[preprocessing guide](preprocessing/README.md) shows how to download them into `data/` or build them yourself.

| Data | `--data-dir` | FID reference |
|---|---|---|
| ImageNet | [`G-REPA/imagenet-latents-*`](https://huggingface.co/G-REPA) | ADM's `VIRTUAL_imagenet256_labeled.npz` (`REFERENCE_NPZ`) |
| MS-COCO | [`SaeedRazavi/mscoco256-t2i-features`](https://huggingface.co/datasets/SaeedRazavi/mscoco256-t2i-features) `features/` | `fid_stats/fid_stats_mscoco256_val.npz` in the same repo (`--ref-stats`) |
| BLIP3o | [`diffusion-bench/blip3o-256`](https://huggingface.co/datasets/diffusion-bench/blip3o-256) | [`SaeedRazavi/blip3o-256-fid50k`](https://huggingface.co/datasets/SaeedRazavi/blip3o-256-fid50k) (`--ref-stats`, `--eval-prompts`) |

## 🚀 3. Training

Set these environment variables for W&B logging (`--report-to none` disables it):

```bash
export ENTITY=<wandb-entity>
export PROJECT=<wandb-project>   # optional: without it, c2i logs to recon-c2i and t2i to recon-t2i
export WANDB_KEY=<key>
```

Checkpoints, logs and evaluation results (`evals.csv`, `evals_t2i.csv`) go to `exps/<exp-name>/`, and a re-run
resumes from the newest checkpoint.

### Class-to-image: SiT on ImageNet

To train SiT-XL/2 with RECON on ImageNet, run:

```bash
REFERENCE_NPZ=data/imagenet/VIRTUAL_imagenet256_labeled.npz \
uv run accelerate launch --num_processes 4 --mixed_precision bf16 c2i_train.py \
    --exp-name sitxl-dinov2b-recon --data-dir data/imagenet
```

<details>
<summary>Options (<code>c2i_train.py</code>)</summary>

- `--model-size`: SiT size, `B`, `L` or `XL` (default)
- `--vae-name`: VAE of the latents in `--data-dir` (see [VAEs](#vaes)), default `sdvae-ft-mse-f8d4`
- `--enc-type`: teacher encoder(s), comma-separated (see [Teacher encoders](#teacher-encoders)), default `dinov2-vit-b`
- `--proj-coeff`: weight of the REPA alignment loss (default 1.5)
- `--encoder-depth`: block at which features are aligned and semantic tokens are injected (default 8, or 4 for `B`)
- `--num-semantic-tokens`: K, a perfect square (default 1). `0` trains plain REPA (use it with `--proj-coeff 0.5`)
- `--sem-proj-type`: `zeropad` (default, no parameters) or `linear` (zero-initialized Linear) head for the tokens
- `--sem-inject-max-t`: during training, tokens are visible only for t ≤ this value (default 0.5). `1.0` keeps them visible at every t
- `--qk-norm`: LayerNorm on queries and keys, as in REPA
- `--class-dropout-prob`: label dropout for classifier-free guidance (default 0.1). `0` trains without a null class
  (no CFG, autoguidance still works)
- `--epochs`, `--batch-size`: training length and global batch size (default 80 and 256)
- `--eval-every`, `--eval-samples`, `--sampler`, `--num-sampling-steps`, `--cfg-scale`: online evaluation (default
  every 100k steps on 50k samples, SDE with 250 steps, no guidance)

</details>

### Text-to-image: MMDiT on MS-COCO

To train MMDiT with RECON on MS-COCO, run:

```bash
uv run accelerate launch --num_processes 4 --mixed_precision fp16 t2i_train.py \
    --exp-name mmdit-dinov2b-recon --data-dir data/mscoco256/features \
    --ref-stats data/mscoco256/fid_stats/fid_stats_mscoco256_val.npz
```

<details>
<summary>Options (<code>t2i_train.py</code>, MMDiT)</summary>

- `--vae-name`: VAE (see [VAEs](#vaes)), default `sdvae-ft-ema-f8d4`. The hosted features fit `sdvae-ft-ema-f8d4`
  and `sdvae-ft-mse-f8d4`. Other VAEs need new features (see the [preprocessing guide](preprocessing/README.md))
- `--enc-type`: teacher encoder(s), comma-separated (see [Teacher encoders](#teacher-encoders)), default `dinov2-vit-b`.
  `none` trains without REPA
- `--proj-coeff`: weight of the REPA alignment loss (default 1.5)
- `--encoder-depth`: block at which features are aligned and semantic tokens are injected (default 8)
- `--num-semantic-tokens`: K, a perfect square (default 1). `0` trains plain REPA (use it with `--proj-coeff 0.5`)
- `--sem-proj-type`: `zeropad` (default, no parameters) or `linear` (zero-initialized Linear) head for the tokens
- `--sem-inject-max-t`: during training, tokens are visible only for t ≤ this value (default 0.5). `1.0` keeps them visible at every t
- `--cfg-prob`: probability of training on the empty caption, for classifier-free guidance (default 0.1)
- `--max-train-steps`, `--batch-size`: training length and global batch size (default 150k and 256)
- `--eval-every`, `--eval-samples`, `--num-sampling-steps`, `--cfg-scale`: online evaluation (default every 50k
  steps on 10,240 captions, 50 steps, no guidance)

</details>

### Text-to-image: DDT on BLIP3o

To train DDT with RECON on BLIP3o, run:

```bash
# run on each of the 4 nodes, with <rank> = 0, 1, 2, 3
uv run accelerate launch --num_processes 16 --num_machines 4 --machine_rank <rank> \
    --main_process_ip <node-0 host> --main_process_port 29500 t2i_train.py --model ddt \
    --exp-name ddt-dinov2b-recon --data-dir data/blip3o-256 \
    --ref-stats data/blip3o-256-fid50k/reference_stats.npz --eval-prompts data/blip3o-256-fid50k/prompts/val
```

<details>
<summary>Options (<code>t2i_train.py --model ddt</code>)</summary>

`--model ddt` sets DiffusionBench's recipe as the defaults (`DDT_DEFAULTS` in `t2i_train.py`).

- `--vae-name`: VAE (see [VAEs](#vaes)), default `sdvae-ft-mse-f8d4`
- `--enc-type`: teacher encoder(s), comma-separated (see [Teacher encoders](#teacher-encoders)), default `dinov2-vit-b`.
  `none` trains plain DDT
- `--proj-coeff`: weight of the REPA alignment loss (default 1.5)
- `--encoder-depth`: block at which features are aligned and semantic tokens are injected (default 8)
- `--num-semantic-tokens`: K, a perfect square (default 1). `0` trains plain REPA (use it with `--proj-coeff 0.5`)
- `--sem-proj-type`: DDT supports only `linear` (zero-initialized Linear) as the head for the tokens
- `--sem-inject-max-t`: during training, tokens are visible only for t ≤ this value (default 0.5). `1.0` keeps them visible at every t
- `--cfg-prob`: probability of training on the empty caption, for classifier-free guidance (default 0.1)
- `--blip3o-splits`: BLIP3o splits to train on (default all three)
- `--max-train-steps`, `--batch-size`: training length and global batch size (default 100k and 1024)
- `--eval-every`, `--eval-samples`, `--num-sampling-steps`, `--cfg-scale`: online evaluation (default every 10k
  steps on 10,000 prompts, 50 steps, no guidance)

</details>

### Teacher encoders and VAEs

<a id="teacher-encoders"></a>
<details>
<summary>Teacher encoders</summary>

`--enc-type` takes `<type>-vit-<config>`. To use several teachers, separate them with commas.

| Type | Configs | Weights |
|---|---|---|
| `dinov2`, `dinov2reg` | `s`, `b`, `l`, `g`. Add `noaffine` (e.g. `bnoaffine`) for a final LayerNorm without affine parameters | torch.hub `facebookresearch/dinov2` |
| `dinov2mixed` | as `dinov2`, with 10% of the CLS token mixed into every patch token | torch.hub `facebookresearch/dinov2` |
| `dinov3` | `s16`, `s16plus`, `b16`, `l16`, `h16plus`, `7b16` | local [DINOv3](https://github.com/facebookresearch/dinov3) repo + checkpoint (see [DINOv3 setup](#dinov3-setup)) |
| `dino` | `s`, `b` | torch.hub `facebookresearch/dino` |
| `pe`, `spatialpe`, `langpe` | `t`/`s`/`b`/`l`/`g` (Core), `b`/`l`/`g` (Spatial), `l`/`g` (Lang). Add `norm` (e.g. `lnorm`) to apply PE's final LayerNorm | Hugging Face `facebook/PE-*` |
| `webssl` | e.g. `dino300m_full2b_224`, `dino1b_full2b_224` | Hugging Face `facebook/webssl-*` |
| `siglip`, `siglip2` | `b`, `l`, `so400m` (+ `g` for SigLIP2) | Hugging Face `google/siglip*` |
| `clip` | `L` | OpenAI `clip` package |
| `mocov3`, `mae`, `jepa` | `s`/`b`/`l`, `l`, `h` | checkpoint in `pretrained_models/` (see [MoCo v3, MAE and I-JEPA weights](#teacher-weights)) |

<a id="dinov3-setup"></a>
DINOv3 loads from `dinov3/` (a clone of the DINOv3 repository) and
`pretrained_models/dinov3_<name>_pretrain_lvd1689m-<hash>.pth`. To use other directories, set `DINOV3_REPO_DIR`
and `DINOV3_CKPT_DIR`. Request the weights on the
[DINOv3 download page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/).

<a id="teacher-weights"></a>
MoCo v3, MAE and I-JEPA load these files from `pretrained_models/` (the file names of REPA):
- `mocov3_vits.pth`, `mocov3_vitb.pth`: ViT-S/16 and ViT-B/16 from [MoCo v3](https://github.com/facebookresearch/moco-v3)
- `mocov3_vitl.pth`: ViT-L/16 from [RCG](https://github.com/LTH14/rcg)
- `mae_vitl.pth`: ViT-L from [MAE](https://github.com/facebookresearch/mae)
- `ijepa_vith.pth`: ViT-H/14 (ImageNet-1K) from [I-JEPA](https://github.com/facebookresearch/ijepa)

</details>

<a id="vaes"></a>
<details>
<summary>VAEs</summary>

`recon/models/vae.py` defines the `--vae-name` keys. The model input size is `resolution / downsampling`, and
the patch size gives the model 16×16 tokens at 256×256.

| `--vae-name` | Weights | Latent grid × channels | Normalization |
|---|---|---|---|
| `sdvae-ft-ema-f8d4` | `stabilityai/sd-vae-ft-ema` | 32×32 × 4 | scalar 0.18215 |
| `sdvae-ft-mse-f8d4` | `stabilityai/sd-vae-ft-mse` | 32×32 × 4 | scalar 0.18215 |
| `e2e-sdvae-f8d4` | `REPA-E/e2e-sdvae-hf` | 32×32 × 4 | per-channel mean / std |
| `invae-f16d32` | `REPA-E/invae-hf` | 16×16 × 32 | per-channel mean / std |
| `vavae-f16d32` | `REPA-E/vavae-hf` | 16×16 × 32 | per-channel mean / std |
| `e2e-vavae-f16d32` | `REPA-E/e2e-vavae-hf` | 16×16 × 32 | per-channel mean / std |

</details>

## 📊 4. Evaluation

To evaluate a checkpoint on its own (the configuration is read from the checkpoint):

### Class-to-image

```bash
# ImageNet: 50k samples, SDE with 250 steps; --cfg-scale with --guidance-low/--guidance-high for guidance
REFERENCE_NPZ=data/imagenet/VIRTUAL_imagenet256_labeled.npz \
uv run accelerate launch --num_processes 4 --mixed_precision bf16 c2i_eval.py \
    --ckpt exps/<exp-name>/checkpoints/ep-00080.pt
```

<details>
<summary>Click to expand for autoguidance parameters</summary>

- `--weak-ckpt`: weak checkpoint for [autoguidance](https://arxiv.org/abs/2406.02507), e.g. the 20-epoch checkpoint `ep-00020.pt`
- `--ag-scale`: autoguidance scale w (default: no autoguidance)
- `--ag-projected`: projected autoguidance. It drops the part of the guidance update that is parallel to the model's
  clean estimate, at each latent position (the projection of [APG](https://arxiv.org/abs/2410.02416))
- `--ste-lambda`: strength λ of Semantic Token Extrapolation (default: off). It moves the semantic token away from the
  token predicted at the first step
- `--guidance-low`, `--guidance-high`: guidance interval (t runs from 1 to 0)

Settings of SiT-XL/2 + RECON (the weak model is an earlier SiT-XL/2 + RECON checkpoint):

| Guidance | Epochs | Weak model | Steps | w | Interval | λ |
|---|---|---|---|---|---|---|
| Autoguidance | 80 | 20 epochs | 50 | 2 | (0, 0.7) | - |
| Improved | 80 | 20 epochs | 50 | 2.5 | (0, 0.7) | 0.05 |
| Autoguidance | 480 | 25 epochs | 100 | 1.4 | (0, 0.85) | - |
| Improved | 480 | 25 epochs | 100 | 2 | (0, 0.85) | 0.05 |

```bash
REFERENCE_NPZ=data/imagenet/VIRTUAL_imagenet256_labeled.npz \
uv run accelerate launch --num_processes 4 --mixed_precision bf16 c2i_eval.py \
    --ckpt exps/<exp-name>/checkpoints/ep-00080.pt --weak-ckpt exps/<exp-name>/checkpoints/ep-00020.pt \
    --num-steps 50 --ag-scale 2.5 --ag-projected --ste-lambda 0.05 --guidance-high 0.7
```

</details>

### Text-to-image

```bash
# MMDiT on 40,192 MS-COCO captions, DDT on the 50k BLIP3o prompts; no guidance unless --cfg-scale > 1
uv run accelerate launch --num_processes 4 t2i_eval.py --ckpt exps/<exp-name>/checkpoints/0150000.pt
```

The text-to-image evaluation reuses the data and FID reference of the training run (override with
`--data-dir`, `--prompts`, `--ref-stats`).

## 💡 Acknowledgements

This codebase uses code from several excellent open-source projects, including:
- [DiffusionBench](https://github.com/End2End-Diffusion/diffusion-bench)
- [REPA](https://github.com/sihyun-yu/REPA)
- [REPA-E](https://github.com/End2End-Diffusion/REPA-E)
- [U-ViT](https://github.com/baofff/U-ViT)
- the official releases of the teacher encoders

We sincerely thank the authors for making their work publicly available.

## 📝 Citation

If you find our work useful, please consider citing:

```bibtex
@misc{razavi2026recon,
  title         = {Representation Conditioning for Generation},
  author        = {Saeed Razavi and Xingjian Leng and Liang Zheng},
  year          = {2026},
  eprint        = {2610.20608},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2610.20608}
}
```
