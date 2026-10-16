"""Evaluate a text-to-image checkpoint: FID and CLIP score for MMDiT (MS-COCO) or DDT (BLIP3o).

MMDiT is based on REPA's generate_t2i.py, and DDT on the BLIP3o FID protocol of DiffusionBench.
REPA: https://github.com/sihyun-yu/REPA
DiffusionBench: https://github.com/End2End-Diffusion/diffusion-bench
"""

import argparse
import os

import torch

from accelerate import Accelerator
from accelerate.utils import set_seed

from recon.data.mscoco import MSCOCO256Features
from recon.eval.t2i import append_eval_csv, evaluate, evaluate_ddt
from recon.models.ddt import DDT
from recon.models.mmdit import MMDiT
from recon.models.text_encoder import Qwen3TextEncoder
from recon.models.vae import HF_VAE_DS_FACTOR, load_vae

# Defaults of the options left unset, per model: REPA's MS-COCO protocol for MMDiT, DiffusionBench's
# BLIP3o FID-50k protocol for DDT
EVAL_DEFAULTS = {
    "mmdit": dict(num_fid_samples=40192, per_proc_batch_size=32, heun=True),
    "ddt": dict(num_fid_samples=50000, per_proc_batch_size=64, heun=False),
}


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    if not torch.cuda.is_available():
        print("WARNING: no GPU available, sampling on CPU (debug only)")
    torch.set_grad_enabled(False)

    accelerator = Accelerator(mixed_precision=None)
    device = accelerator.device
    if args.global_seed is not None:
        set_seed(args.global_seed + accelerator.process_index)

    # The model config, including the semantic tokens, comes from the checkpoint's training args
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt[args.ckpt_key]
    step = ckpt.get("steps", 0)
    train_args = ckpt.get("args", None)

    def _ta(name, default):
        return getattr(train_args, name, default) if train_args is not None else default

    model_type = _ta("model", "mmdit")
    for name, value in EVAL_DEFAULTS[model_type].items():  # options left unset follow the model's protocol
        if getattr(args, name) is None:
            setattr(args, name, value)
    # the data and FID reference the checkpoint was trained and evaluated with, unless given
    for name, train_name in (("data_dir", "data_dir"), ("ref_stats", "ref_stats"), ("prompts", "eval_prompts")):
        if getattr(args, name) is None:
            setattr(args, name, _ta(train_name, None))
    needed = ("data_dir", "ref_stats") if model_type == "mmdit" else ("prompts", "ref_stats")
    missing = [f"--{name.replace('_', '-')}" for name in needed if getattr(args, name) is None]
    assert not missing, f"pass {' and '.join(missing)} (not in the checkpoint's training args)"
    assert model_type == "mmdit" or args.mode == "ode", "DDT samples with the Euler ODE sampler (--mode ode)"

    sem_cfg = dict(
        num_semantic_tokens=_ta("num_semantic_tokens", 4),
        sem_inject_max_t=_ta("sem_inject_max_t", 1.0),
    )
    if model_type == "mmdit":
        sem_cfg["sem_proj_type"] = _ta("sem_proj_type", "linear")
    if accelerator.is_main_process:
        print(f"Semantic-token config from checkpoint args: {sem_cfg}")

    # the training VAE and resolution, unless given; the model is sized from its latents as in t2i_train.py
    vae_name = args.vae_name or _ta("vae_name", "sdvae-ft-ema-f8d4")
    args.resolution = args.resolution or _ta("resolution", 256)
    vae, latents_scale, latents_bias = load_vae(vae_name, device)
    ds_factor = HF_VAE_DS_FACTOR[vae_name]
    latent_cfg = dict(input_size=args.resolution // ds_factor, patch_size=16 // ds_factor,
                      in_channels=vae.config.latent_channels)
    # the teacher widths come from the weights: the output size of each REPA projector
    num_projectors = len({k.split(".")[1] for k in state_dict if k.startswith("projectors.")})
    last_layer = "4.weight" if model_type == "mmdit" else "weight"
    z_dims = [state_dict[f"projectors.{i}.{last_layer}"].shape[0] for i in range(num_projectors)]
    if model_type == "mmdit":
        model = MMDiT(
            z_dims=z_dims,
            encoder_depth=_ta("encoder_depth", args.encoder_depth),
            **latent_cfg,
            **sem_cfg,
        )
    else:  # the text-encoder width also comes from the weights
        model = DDT(
            z_dims=z_dims,
            context_dim=state_dict["context_embedder.proj.weight"].shape[1],
            encoder_depth=_ta("encoder_depth", args.encoder_depth),
            **latent_cfg,
            **sem_cfg,
        )
    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    del ckpt
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"

    # Setup data:
    if model_type == "mmdit":
        dataset = MSCOCO256Features(path=args.data_dir, mode='val', ret_caption=True)
        assert args.num_fid_samples <= len(dataset.test)
        if accelerator.is_main_process:
            print(f"Loaded MS-COCO features from {args.data_dir} ({len(dataset.test):,} val images)")
    else:
        from datasets import load_from_disk
        prompts = load_from_disk(args.prompts)["text"]
        assert args.num_fid_samples <= len(prompts)

    # Create folder to save samples:
    ckpt_tag = os.path.splitext(os.path.basename(args.ckpt))[0]
    folder_name = f"{args.prefix + '-' if args.prefix else ''}{'coco' if model_type == 'mmdit' else 'blip3o'}-" \
                  f"{ckpt_tag}-{args.ckpt_key}-size-{args.resolution}-vae-{vae_name}-" \
                  f"cfg-{args.cfg_scale}-steps-{args.num_steps}-{'heun-' if args.heun else ''}seed-{args.global_seed}-{args.mode}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if accelerator.is_main_process:
        if model_type == "mmdit":
            print(f"Saving .png samples at {sample_folder_dir}")
        print(f"Total number of images that will be sampled: {args.num_fid_samples}")
        print(f"{type(model).__name__} Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if model_type == "mmdit":
        fid, clip_sim = evaluate(
            model, vae, latents_scale, latents_bias, accelerator, dataset.test, dataset.empty_context,
            args.ref_stats, args.num_fid_samples, args.per_proc_batch_size,
            mode=args.mode, heun=args.heun, num_steps=args.num_steps, cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low, guidance_high=args.guidance_high,
            clip_model=args.clip_model if args.clip_sim else None,
            sample_dir=sample_folder_dir, save_npz=args.save_npz, fid_batch_size=args.fid_batch_size,
        )
    else:
        fid, clip_sim = evaluate_ddt(
            model, vae, latents_scale, latents_bias, accelerator, Qwen3TextEncoder(device),
            prompts[:args.num_fid_samples], args.ref_stats, shard_dir=args.sample_dir,
            batch_size=args.per_proc_batch_size, num_steps=args.num_steps, cfg_scale=args.cfg_scale,
            clip_model=args.clip_model if args.clip_sim else None,
            npz_path=f"{sample_folder_dir}.npz" if args.save_npz else None, fid_batch_size=args.fid_batch_size,
        )
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"  ckpt: {args.ckpt} [{args.ckpt_key}] step={step}")
        print(f"  mode={args.mode} heun={args.heun} steps={args.num_steps} cfg={args.cfg_scale}")
        print(f"  num_samples={args.num_fid_samples}")
        print(f"  FID:      {fid:.4f}")
        print(f"  CLIP-sim: {clip_sim:.4f}")
        print(f"{'='*60}\n")

        csv_path = os.path.join(os.path.dirname(os.path.dirname(args.ckpt)), "evals_t2i.csv")
        append_eval_csv(csv_path, step, fid, clip_sim, ckpt_key=args.ckpt_key, mode=args.mode,
                        heun=args.heun, num_steps=args.num_steps, cfg_scale=args.cfg_scale,
                        num_samples=args.num_fid_samples)
        print(f"Appended results to {csv_path}")
        print("Done.")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving:
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a t2i_train.py checkpoint.")
    parser.add_argument("--ckpt-key", type=str, choices=["ema", "model"], default="ema",
                        help="Reported numbers use the EMA weights.")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--save-npz", action=argparse.BooleanOptionalAction, default=True,
                        help="Also write an ADM-style samples npz next to the sample folder.")

    # model
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=None,
                        help="default: the checkpoint's --resolution")

    # vae
    parser.add_argument("--vae-name", type=str, default=None,
                        help="VAE that decodes the samples (default: the checkpoint's --vae-name)")

    # dataset / reference (unset: the checkpoint's training args; other unset options: EVAL_DEFAULTS)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="MMDiT: MS-COCO features (default: the checkpoint's --data-dir)")
    parser.add_argument("--prompts", type=str, default=None,
                        help="DDT: the prompts of the frozen reference (build_blip3o_reference.py; "
                             "default: the checkpoint's --eval-prompts)")
    parser.add_argument("--ref-stats", type=str, default=None,
                        help="FID reference statistics (MMDiT: U-ViT's fid_stats_mscoco256_val.npz; "
                             "DDT: reference_stats.npz of build_blip3o_reference.py; default: the checkpoint's)")

    # number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=None)
    parser.add_argument("--num-fid-samples", type=int, default=None, help="40192 (MMDiT) or 50000 (DDT)")
    parser.add_argument("--fid-batch-size", type=int, default=128)

    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="ode", choices=["ode", "sde"])
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="1.0 = no guidance")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=None,
                        help="MMDiT ODE only (default on)")
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)

    # metrics
    parser.add_argument("--clip-sim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-large-patch14")

    args = parser.parse_args()
    main(args)
