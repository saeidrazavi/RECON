"""Evaluate a class-to-image checkpoint: FID, Inception Score, precision and recall on ImageNet 256."""

import argparse
import os

import torch

from recon.eval.c2i import append_eval_csv, evaluate, load_model_and_vae, make_accelerator, require_reference_npz


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed FID evaluation for a single checkpoint")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint .pt (step is read from filename)")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name for CSV output (default: inferred from ckpt path)")
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--guidance-low", type=float, default=0.0,
                        help="Lower time bound for CFG or autoguidance (time runs 1->0).")
    parser.add_argument("--guidance-high", type=float, default=1.0,
                        help="Upper time bound for CFG or autoguidance, a.k.a. t_max_cfg. Guidance is "
                             "applied only for t in [guidance_low, guidance_high].")
    parser.add_argument("--weak-ckpt", type=str, default=None,
                        help="Weak checkpoint for autoguidance, e.g. the 20-epoch checkpoint (ep-00020.pt).")
    parser.add_argument("--ag-scale", type=float, default=None,
                        help="Autoguidance scale w (needs --weak-ckpt). Default: no autoguidance.")
    parser.add_argument("--ag-projected", action="store_true",
                        help="Projected autoguidance: drop the part of the guidance update that is parallel to the "
                             "model's clean estimate, at each latent position.")
    parser.add_argument("--ste-lambda", type=float, default=None,
                        help="Strength of Semantic Token Extrapolation, e.g. 0.05. Default: off.")
    parser.add_argument("--sampler", type=str, default="sde", choices=["ode", "sde"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (args.weak_ckpt is None) != (args.ag_scale is None):
        parser.error("--weak-ckpt and --ag-scale go together")
    if args.ag_projected and args.ag_scale is None:
        parser.error("--ag-projected needs --ag-scale")
    if (args.ag_scale is not None or args.ste_lambda is not None) and args.cfg_scale > 1.0:
        parser.error("use either --cfg-scale or autoguidance / --ste-lambda")

    require_reference_npz()

    # Infer exp_name: .../exps/<exp_name>/checkpoints/ep-00005.pt -> <exp_name>
    if args.exp_name is None:
        args.exp_name = os.path.basename(os.path.dirname(os.path.dirname(args.ckpt)))

    accelerator = make_accelerator(mixed_precision="bf16")
    device = accelerator.device

    torch.manual_seed(args.seed + accelerator.process_index)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Build model + VAE from checkpoint's saved training args
    model, vae, latents_scale, latents_bias, step, train_args = \
        load_model_and_vae(args.ckpt, device, accelerator)
    weak_model = load_model_and_vae(args.weak_ckpt, device, accelerator)[0] if args.weak_ckpt else None
    if args.cfg_scale > 1.0 and model.y_embedder.dropout_prob == 0:
        parser.error("this checkpoint was trained without label dropout (no null class): use autoguidance, "
                     "not --cfg-scale")

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"  OFFLINE EVAL: {args.exp_name}  step={step}")
        print(f"{'='*60}")
        print(f"  ckpt:           {args.ckpt}")
        if weak_model is not None:
            print(f"  weak ckpt:      {args.weak_ckpt}")
        print(f"  num_samples:    {args.num_samples}")
        print(f"  num_gpus:       {accelerator.num_processes}")
        print(f"{'='*60}\n")

    # Temp dir for shards next to the checkpoint
    eval_shard_dir = os.path.join(os.path.dirname(args.ckpt), "..", "_eval_shards")

    fid, is_mean, precision, recall = evaluate(
        model, vae, latents_scale, latents_bias, accelerator,
        step, eval_shard_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        cfg_scale=args.cfg_scale,
        num_steps=args.num_steps,
        sampler=args.sampler,
        guidance_low=args.guidance_low,
        guidance_high=args.guidance_high,
        weak_model=weak_model,
        ag_scale=args.ag_scale,
        ag_projected=args.ag_projected,
        ste_lambda=args.ste_lambda,
    )

    if accelerator.is_main_process:
        sampler_tag = "sde" if args.sampler == "sde" else "ode"
        csv_dir = f"evals_offline/{sampler_tag}_step{args.num_steps}_n{args.num_samples}"
        if args.ag_scale is not None:  # guided runs get their own folder
            csv_dir += f"_ag{args.ag_scale}{'proj' if args.ag_projected else ''}_t{args.guidance_low}-{args.guidance_high}"
        if args.ste_lambda is not None:
            csv_dir += f"_ste{args.ste_lambda}"
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f"{args.exp_name}.csv")
        append_eval_csv(csv_path, step, {"fid": fid, "is": is_mean, "precision": precision, "recall": recall})

        msg = f"\n  step={step}  FID={fid:.4f}  IS={is_mean:.4f}"
        if precision is not None:
            msg += f"  P={precision:.4f}  R={recall:.4f}"
        print(f"{msg}  -> {csv_path}")

    accelerator.wait_for_everyone()
