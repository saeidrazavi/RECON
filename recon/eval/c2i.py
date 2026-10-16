"""Class-to-image evaluation: distributed sampling with the EMA model and FID, Inception Score,
precision and recall against ADM's reference batch (REFERENCE_NPZ)."""

import csv
import math
import os
from datetime import timedelta

import numpy as np
import torch
import scipy.linalg
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

from recon.diffusion.samplers import euler_sampler, euler_maruyama_sampler

# ADM's ImageNet 256 reference batch (VIRTUAL_imagenet256_labeled.npz), see the README
REFERENCE_NPZ = os.getenv("REFERENCE_NPZ")

# Rank 0 computes the metrics while the other ranks wait, which takes longer than the
# default NCCL timeout (600 s) for a 50k-sample eval.
DIST_TIMEOUT_HOURS = float(os.getenv("DIST_TIMEOUT_HOURS", "4"))


EVAL_CSV_FIELDS = ["step", "fid", "is", "precision", "recall"]


def append_eval_csv(csv_path, step, metrics):
    """Append one row to csv_path: the step and every metric that is not None (4 decimals)."""
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({"step": step, **{k: f"{v:.4f}" for k, v in metrics.items() if v is not None}})


def require_reference_npz():
    """Stop before training or sampling if REFERENCE_NPZ does not point at a file."""
    if not (REFERENCE_NPZ and os.path.isfile(REFERENCE_NPZ)):
        raise SystemExit(f"REFERENCE_NPZ={REFERENCE_NPZ!r}: set it to ADM's VIRTUAL_imagenet256_labeled.npz "
                         "(see the README)")


def make_accelerator(**kwargs):
    """`Accelerator(**kwargs)` with a distributed timeout of DIST_TIMEOUT_HOURS."""
    handlers = list(kwargs.pop("kwargs_handlers", None) or [])
    if not any(isinstance(h, InitProcessGroupKwargs) for h in handlers):
        handlers.append(InitProcessGroupKwargs(timeout=timedelta(hours=DIST_TIMEOUT_HOURS)))
    return Accelerator(kwargs_handlers=handlers, **kwargs)


def load_model_and_vae(ckpt, device, accelerator):
    """Rebuild the (EMA) model + VAE from a training checkpoint.

    Returns ``(model, vae, latents_scale, latents_bias, step, train_args)``.
    Used by the offline single-checkpoint eval below.
    """
    from recon.encoders.vision_encoder import load_encoders
    from recon.models.sit import build_sit, MODEL_SIZES
    from recon.models.vae import HF_VAE_DS_FACTOR, load_vae

    state_dict = torch.load(ckpt, map_location=device, weights_only=False)
    train_args = state_dict["args"]
    step = state_dict.get("steps", 0)

    encoders = load_encoders(train_args.enc_type, device, 256, accelerator)

    vae_name = train_args.vae_name
    vae, latents_scale, latents_bias = load_vae(vae_name, device)

    overrides = MODEL_SIZES[train_args.model_size].copy() if train_args.model_size is not None else {}
    overrides["input_size"] = 256 // HF_VAE_DS_FACTOR[vae_name]
    overrides["patch_size"] = overrides["input_size"] // 16
    overrides["in_channels"] = vae.config.latent_channels
    overrides["z_dims"] = [enc.embed_dim for enc in encoders]
    overrides["num_semantic_tokens"] = getattr(train_args, "num_semantic_tokens", 4)
    overrides["qk_norm"] = getattr(train_args, "qk_norm", False)
    overrides["sem_proj_type"] = getattr(train_args, "sem_proj_type", "linear")
    overrides["class_dropout_prob"] = getattr(train_args, "class_dropout_prob", 0.1)  # 0: no null class
    # encoder_depth changes no parameter shape, so load_state_dict cannot detect a wrong value
    if getattr(train_args, "encoder_depth", None) is not None:
        overrides["encoder_depth"] = train_args.encoder_depth

    model = build_sit(**overrides).to(device)
    model.load_state_dict(state_dict.get("ema", state_dict))
    model.eval()

    return model, vae, latents_scale, latents_bias, step, train_args


def _fid_from_moments(mu1, sigma1, mu2, sigma2):
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)

    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(max(fid, 0.0))


@torch.no_grad()
def _inception_features(images_nhwc, batch_size, device):
    """Compute Inception 2048-d features and logits in a single pass."""
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

    x = np.transpose(images_nhwc, (0, 3, 1, 2))
    xt = torch.from_numpy(x).to(device=device, dtype=torch.uint8)

    fe = FeatureExtractorInceptionV3(
        name="inception-v3-compat", features_list=["2048", "logits_unbiased"]
    ).to(device).eval()

    feats_2048 = []
    feats_logits = []
    for i in range(0, xt.shape[0], batch_size):
        batch = xt[i:i + batch_size]
        f_2048, f_logits = fe(batch)
        feats_2048.append(f_2048.detach().cpu())
        feats_logits.append(f_logits.detach().cpu())

    del fe
    torch.cuda.empty_cache()

    feats_2048 = torch.cat(feats_2048, dim=0)
    feats_logits = torch.cat(feats_logits, dim=0)
    return feats_2048.double().numpy(), feats_logits


def _inception_score(logits, num_splits=10):
    """Compute Inception Score (mean) from raw logits."""
    probs = torch.nn.functional.softmax(logits.float(), dim=1).numpy()
    n = len(probs)
    split_size = n // num_splits
    scores = []
    for i in range(num_splits):
        part = probs[i * split_size : (i + 1) * split_size]
        p_y = part.mean(axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
        scores.append(float(np.exp(kl.sum(axis=1).mean())))
    return float(np.mean(scores))


@torch.no_grad()
def _compute_prec_recall(ref_feats, gen_feats, k=3, batch_size=1000, device="cuda"):
    """Compute precision and recall via k-NN manifold estimation."""
    ref = torch.from_numpy(ref_feats.astype(np.float32)).to(device)
    gen = torch.from_numpy(gen_feats.astype(np.float32)).to(device)

    def manifold_radii(feats):
        radii = torch.zeros(len(feats), device=device)
        for i in range(0, len(feats), batch_size):
            batch = feats[i:i + batch_size]
            D = torch.cdist(batch, feats)
            # k+1 to skip self (distance 0)
            radii[i:i + len(batch)] = torch.kthvalue(D, k + 1, dim=1).values
        return radii

    ref_radii = manifold_radii(ref)
    gen_radii = manifold_radii(gen)

    # Precision: fraction of gen samples inside ref manifold
    prec = 0
    for i in range(0, len(gen), batch_size):
        batch = gen[i:i + batch_size]
        D = torch.cdist(batch, ref)
        prec += (D <= ref_radii.unsqueeze(0)).any(dim=1).sum().item()

    # Recall: fraction of ref samples inside gen manifold
    rec = 0
    for i in range(0, len(ref), batch_size):
        batch = ref[i:i + batch_size]
        D = torch.cdist(batch, gen)
        rec += (D <= gen_radii.unsqueeze(0)).any(dim=1).sum().item()

    del ref, gen, ref_radii, gen_radii
    torch.cuda.empty_cache()

    return float(prec / len(gen_feats)), float(rec / len(ref_feats))


def compute_metrics(gen_images, ref_stats, batch_size=128, device="cuda"):
    """Compute FID, IS, precision, and recall."""
    gen_feats, logits = _inception_features(gen_images, batch_size, device)

    mu_gen = gen_feats.mean(axis=0)
    sigma_gen = np.cov(gen_feats, rowvar=False)

    if "mu" in ref_stats and "sigma" in ref_stats:
        mu_ref, sigma_ref = ref_stats["mu"], ref_stats["sigma"]
    elif "ref_mu" in ref_stats and "ref_sigma" in ref_stats:
        mu_ref, sigma_ref = ref_stats["ref_mu"], ref_stats["ref_sigma"]
    else:
        raise KeyError(f"Reference has no mu/sigma keys, found: {list(ref_stats.keys())}")

    fid = _fid_from_moments(mu_gen, sigma_gen, mu_ref, sigma_ref)
    is_mean = _inception_score(logits)

    # Precision & Recall from reference images in the NPZ
    precision, recall = None, None
    if "arr_0" in ref_stats:
        ref_feats, _ = _inception_features(ref_stats["arr_0"], batch_size, device)
        precision, recall = _compute_prec_recall(ref_feats, gen_feats, device=device)

    return fid, is_mean, precision, recall


@torch.no_grad()
def evaluate(
    ema_model, vae, latents_scale, latents_bias, accelerator,
    global_step, save_dir,
    num_samples=10_000, batch_size=64, cfg_scale=1.0,
    num_steps=50, sampler="ode",
    guidance_low=0.0, guidance_high=1.0,
    weak_model=None, ag_scale=None, ag_projected=False, ste_lambda=None,
):
    """Distributed generation + metrics. Returns (FID, IS, precision, recall) on rank 0.

    CFG, or autoguidance with weak_model, is applied while guidance_low <= t <= guidance_high (t runs from 1
    to 0). ag_projected and ste_lambda select projected autoguidance and Semantic Token Extrapolation.
    """
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    in_ch = ema_model.in_channels
    lat_size = ema_model.x_embedder.img_size[0]
    samples_per_gpu = math.ceil(num_samples / world_size)

    ema_model.eval()
    generations = []

    iters = math.ceil(samples_per_gpu / batch_size)
    it = tqdm(range(iters), desc=f"[Rank {rank}] Eval") if rank == 0 else range(iters)

    for _ in it:
        n = min(batch_size, samples_per_gpu - len(generations))
        if n <= 0:
            break

        z = torch.randn(n, in_ch, lat_size, lat_size, device=device)
        y = torch.randint(0, 1000, (n,), device=device)
        y_null = torch.full_like(y, 1000)  # null class label for classifier-free guidance

        sample_fn = euler_maruyama_sampler if sampler == "sde" else euler_sampler
        samples = sample_fn(
            ema_model, z, {"y": y}, {"y": y_null}, num_steps=num_steps,
            cfg_scale=cfg_scale,
            guidance_low=guidance_low, guidance_high=guidance_high,
            weak_model=weak_model, ag_scale=ag_scale, ag_projected=ag_projected, ste_lambda=ste_lambda,
        ).to(torch.float32)
        samples = samples / latents_scale + latents_bias
        decoded = vae.decode(samples).sample
        decoded = ((decoded + 1) / 2).clamp(0, 1)
        imgs = decoded.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        for img in imgs:
            generations.append(img)

    generations = np.stack(generations[:samples_per_gpu])

    # Save shard per rank
    eval_dir = os.path.join(save_dir, f"eval_{global_step:07d}")
    if accelerator.is_main_process:
        os.makedirs(eval_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    np.savez(os.path.join(eval_dir, f"shard_{rank:02d}.npz"), arr_0=generations)
    accelerator.wait_for_everyone()

    # Rank 0 gathers shards and computes metrics
    fid, is_mean, precision, recall = None, None, None, None
    if accelerator.is_main_process:
        all_imgs = []
        for r in range(world_size):
            shard = np.load(os.path.join(eval_dir, f"shard_{r:02d}.npz"))["arr_0"]
            all_imgs.append(shard)
        all_imgs = np.concatenate(all_imgs)[:num_samples]

        print(f"\n{'='*60}")
        print(f"  EVAL RESULTS — Step {global_step}")
        print(f"{'='*60}")
        print(f"  num_samples:    {len(all_imgs)}")
        print(f"  num_steps:      {num_steps}")
        print(f"  cfg_scale:      {cfg_scale}")
        print(f"  guidance_range: [{guidance_low}, {guidance_high}]")
        if ag_scale is not None:
            print(f"  autoguidance:   w={ag_scale}{' (projected)' if ag_projected else ''}")
        if ste_lambda is not None:
            print(f"  STE lambda:     {ste_lambda}")
        print(f"  reference:      {REFERENCE_NPZ}")

        ref_stats = np.load(REFERENCE_NPZ)
        fid, is_mean, precision, recall = compute_metrics(
            all_imgs, ref_stats, device=str(device),
        )

        print(f"  FID:            {fid:.4f}")
        print(f"  IS:             {is_mean:.4f}")
        if precision is not None:
            print(f"  Precision:      {precision:.4f}")
            print(f"  Recall:         {recall:.4f}")
        print(f"{'='*60}\n")

        # Cleanup shards
        for r in range(world_size):
            p = os.path.join(eval_dir, f"shard_{r:02d}.npz")
            if os.path.exists(p):
                os.remove(p)
        try:
            os.rmdir(eval_dir)
        except OSError:
            pass

    accelerator.wait_for_everyone()
    return fid, is_mean, precision, recall
