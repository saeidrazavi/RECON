import argparse
import re
from copy import deepcopy
import os
from pathlib import Path
import json

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from recon.data.imagenet import HFImgLatentDataset
from recon.diffusion.loss import SILoss
from recon.diffusion.samplers import euler_sampler
from recon.encoders.vision_encoder import load_encoders
from recon.eval.c2i import append_eval_csv, evaluate, make_accelerator, require_reference_npz
from recon.models.sit import build_sit, MODEL_SIZES, SiTConfig
from recon.models.vae import HF_VAE_MAPPER, HF_VAE_DS_FACTOR, load_vae
from recon.utils import array2grid, create_logger, sample_posterior, update_ema
import wandb

logger = get_logger(__name__)


#################################################################################
#                                 Training Loop                                 #
#################################################################################

def find_resume_checkpoint(checkpoint_dir):
    """Path of the newest ep-XXXXX.pt checkpoint in checkpoint_dir, or None."""
    if not os.path.isdir(checkpoint_dir):
        return None
    ckpts = [f for f in os.listdir(checkpoint_dir) if re.fullmatch(r"ep-\d+\.pt", f)]
    return os.path.join(checkpoint_dir, max(ckpts, key=lambda f: int(f[3:-3]))) if ckpts else None


def main(args):
    require_reference_npz()  # FID is evaluated during and at the end of training

    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = make_accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=accelerator_project_config,
    )

    save_dir = os.path.join(args.output_dir, args.exp_name)
    checkpoint_dir = f"{save_dir}/checkpoints"
    eval_csv_path = os.path.join(save_dir, "evals.csv")
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(save_dir, "args.json"), 'w') as f:
            json.dump(vars(args), f, indent=4)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Setup REPA target encoder
    encoders = load_encoders(args.enc_type, device, 256, accelerator)

    # Setup VAE
    vae, latents_scale, latents_bias = load_vae(args.vae_name, device)
    if accelerator.is_main_process:
        def _summary(t):
            if torch.is_tensor(t):
                return f"tensor shape={tuple(t.shape)} dtype={t.dtype} min={t.min().item():.4g} max={t.max().item():.4g} mean={t.float().mean().item():.4g}"
            return f"scalar {t}"
        logger.info(f"[VAE] {args.vae_name}: {type(vae).__name__} from {HF_VAE_MAPPER[args.vae_name]}")
        logger.info(f"[VAE] latents_scale {_summary(latents_scale)}")
        logger.info(f"[VAE] latents_bias  {_summary(latents_bias)}")

    # Create overrides for model based on model size and VAE name
    overrides = MODEL_SIZES[args.model_size].copy() if args.model_size is not None else {}
    overrides["encoder_depth"] = args.encoder_depth   # resolved in parse_args()
    overrides["input_size"] = 256 // HF_VAE_DS_FACTOR[args.vae_name]
    overrides["patch_size"] = overrides["input_size"] // 16
    overrides["in_channels"] = vae.config.latent_channels
    overrides["z_dims"] = [enc.embed_dim for enc in encoders]
    overrides["num_semantic_tokens"] = args.num_semantic_tokens
    overrides["sem_inject_max_t"] = args.sem_inject_max_t
    overrides["sem_proj_type"] = args.sem_proj_type
    overrides["qk_norm"] = args.qk_norm
    overrides["class_dropout_prob"] = args.class_dropout_prob

    model = build_sit(**overrides)
    model = model.to(device)

    # Compute GFLOPs on main process before prepare()
    model_gflops = None
    if accelerator.is_main_process:
        from torch.utils.flop_counter import FlopCounterMode
        model.eval()
        _x = torch.randn(1, overrides["in_channels"], overrides["input_size"], overrides["input_size"], device=device)
        _t = torch.rand(1, device=device)
        _y = torch.randint(0, 1000, (1,), device=device)
        _fc = FlopCounterMode(display=False)
        with _fc:
            model(_x, _t, _y)
        model_gflops = _fc.get_total_flops() / 1e9
        model.train()
        del _x, _t, _y, _fc

    ema = deepcopy(model).to(device)
    ema.requires_grad_(False)

    # create loss function
    loss_fn = SILoss()
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"SiT GFLOPs: {model_gflops:.2f}")

    # Setup optimizer
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Setup data:
    train_dataset = HFImgLatentDataset(args.vae_name, args.data_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # resume: auto-find latest checkpoint
    global_step = 0
    start_epoch = 0
    grad_norm = torch.tensor(0.0, device=device)
    resume_ckpt = find_resume_checkpoint(checkpoint_dir)
    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']
        start_epoch = ckpt['epoch']
        if accelerator.is_main_process:
            logger.info(f"Resumed from {resume_ckpt} (epoch {start_epoch}, step {global_step})")

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    loss_fn = torch.compile(loss_fn)

    if accelerator.is_main_process:
        if args.report_to == "wandb" and "WANDB_KEY" in os.environ:
            wandb.login(key=os.environ["WANDB_KEY"])
        accelerator.init_trackers(
            project_name=os.environ.get("PROJECT", "recon-c2i"),
            config=dict(vars(args)),
            init_kwargs={"wandb": {"name": args.exp_name, "entity": os.environ.get("ENTITY")}},
        )

    progress_bar = tqdm(
        desc="Steps",
        initial=global_step,
        disable=not accelerator.is_main_process,
        miniters=args.log_every,
    )

    # Fixed batch used for periodic sample logging to wandb
    sample_noise = sample_labels = gt_latents = None
    if args.sampling_steps > 0:
        sample_batch_size = max(1, args.n_samples // accelerator.num_processes)
        _raw, gt_x, gt_y = next(iter(train_dataloader))
        gt_x = gt_x[:sample_batch_size].squeeze(dim=1).to(device)
        sample_labels = gt_y[:sample_batch_size].to(device)
        gt_latents = sample_posterior(gt_x, latents_scale=latents_scale, latents_bias=latents_bias)
        sample_noise = torch.randn_like(gt_latents)

    def run_eval():
        """Sample with the EMA model, compute metrics on rank 0, append them to evals.csv."""
        model.eval()
        accelerator.wait_for_everyone()
        fid, is_mean, precision, recall = evaluate(
            ema, vae, latents_scale, latents_bias, accelerator,
            global_step, save_dir,
            num_samples=args.eval_samples,
            batch_size=args.eval_batch_size,
            cfg_scale=args.cfg_scale,
            num_steps=args.num_sampling_steps,
            sampler=args.sampler,
        )
        if accelerator.is_main_process:
            metrics = {"fid": fid, "is": is_mean, "precision": precision, "recall": recall}
            metrics = {k: v for k, v in metrics.items() if v is not None}
            append_eval_csv(eval_csv_path, global_step, metrics)
            if metrics:
                accelerator.log(metrics, step=global_step)
            logger.info(f"Eval results (step {global_step}) appended to {eval_csv_path}")
        model.train()

    def save_checkpoint(epoch):
        """Rank 0 only: save model, EMA and optimizer state to checkpoints/ep-{epoch:05d}.pt."""
        checkpoint = {
            "model": accelerator.unwrap_model(model).state_dict(),
            "ema": ema.state_dict(),
            "opt": optimizer.state_dict(),
            "args": args,
            "steps": global_step,
            "epoch": epoch,
        }
        checkpoint_path = f"{checkpoint_dir}/ep-{epoch:05d}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path} (epoch {epoch}, step {global_step})")

    reached_max_steps = False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for raw_image, x, y in train_dataloader:
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                with accelerator.autocast():
                    zs = [enc.forward_features(enc.preprocess(raw_image))["x_norm_patchtokens"] for enc in encoders]

            with accelerator.accumulate(model):
                model_kwargs = dict(y=y, teacher_zs=zs)
                loss, proj_loss = loss_fn(model, x, model_kwargs, zs=zs)
                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean()
                loss = loss_mean + proj_loss_mean * args.proj_coeff

                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model, decay=args.ema_decay)

            if not accelerator.sync_gradients:
                continue  # still accumulating: no optimizer step was taken
            global_step += 1

            # Periodic eval (step-based)
            if args.eval_every > 0 and global_step % args.eval_every == 0:
                run_eval()

            # Periodic sample logging to wandb
            if (
                args.sampling_steps > 0
                and sample_noise is not None
                and (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0))
            ):
                accelerator.wait_for_everyone()
                with torch.no_grad():
                    samples = euler_sampler(
                        ema, sample_noise,
                        {"y": sample_labels}, {"y": torch.full_like(sample_labels, 1000)},  # 1000 = null class
                        num_steps=args.num_sampling_steps,
                        cfg_scale=args.sample_cfg_scale,
                        guidance_low=0., guidance_high=1.,
                        heun=False,
                    ).to(torch.float32)
                    samples_dec = samples / latents_scale + latents_bias
                    gt_dec = gt_latents / latents_scale + latents_bias
                    samples_dec = vae.decode(samples_dec.to(vae.dtype)).sample
                    gt_dec = vae.decode(gt_dec.to(vae.dtype)).sample
                    samples_dec = ((samples_dec + 1) / 2).clamp(0, 1)
                    gt_dec = ((gt_dec + 1) / 2).clamp(0, 1)
                out_samples = accelerator.gather(samples_dec.to(torch.float32))
                out_gt = accelerator.gather(gt_dec.to(torch.float32))
                if accelerator.is_main_process and args.report_to == "wandb":
                    accelerator.log(
                        {
                            "samples": wandb.Image(array2grid(out_samples)),
                            "gt_samples": wandb.Image(array2grid(out_gt)),
                        },
                        step=global_step,
                    )
                    logger.info(f"Logged {out_samples.size(0)} samples to wandb (step {global_step})")

            if global_step % args.log_every == 0:
                logs = {
                    "loss": accelerator.gather(loss_mean).mean().detach().item(),
                    "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                    "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
                }
                progress_bar.set_postfix(**logs, refresh=False)
                progress_bar.update(args.log_every)
                accelerator.log(logs, step=global_step)
                if accelerator.is_main_process:
                    logger.info(
                        f"step={global_step}  epoch={epoch}  loss={logs['loss']:.4f}  "
                        f"proj_loss={logs['proj_loss']:.4f}  grad_norm={logs['grad_norm']:.4f}"
                    )

            if args.max_steps > 0 and global_step >= args.max_steps:
                reached_max_steps = True
                if accelerator.is_main_process:
                    logger.info(f"Reached max_steps={args.max_steps} at step {global_step}; stopping early.")
                break

        if reached_max_steps:
            break

        # End of epoch: checkpoint (epoch+1 = epochs trained so far)
        if accelerator.is_main_process and (epoch + 1) % args.checkpointing_epochs == 0:
            save_checkpoint(epoch + 1)

    # Final checkpoint and eval, unless the loop already did them at this point
    if accelerator.is_main_process and (reached_max_steps or args.epochs % args.checkpointing_epochs != 0):
        save_checkpoint(args.epochs)
    if not (args.eval_every > 0 and global_step % args.eval_every == 0):
        run_eval()
    if accelerator.is_main_process:
        logger.info(f"Training complete at epoch {args.epochs}, step {global_step}")

    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--checkpointing-epochs", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--report-to", type=str, default="wandb",
                        help="Tracker backend for accelerate (e.g. 'wandb', 'tensorboard', or 'none').")

    # dataset
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with imagenet-latents-images/ and imagenet-latents-<vae-name>/ (see the README)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--vae-name", type=str, default="sdvae-ft-mse-f8d4")

    # precision
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after this many optimizer steps (0 = disabled). "
                             "Runs the final checkpoint + eval on exit. Intended for sanity checks.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--log-every", type=int, default=100, help="Log and update progress bar every N steps")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=16)

    # loss
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=1.5)

    # model
    parser.add_argument("--model-size", type=str, default="XL", choices=["B", "L", "XL"],
                        help="SiT size: B, L or XL (MODEL_SIZES in recon/models/sit.py).")
    parser.add_argument("--encoder-depth", type=int, default=None,
                        help="Block index (1-based) at which REPA reads z_tilde AND the semantic tokens are "
                             "injected -- the two happen at the same block. None (default) uses the value the "
                             "model size implies (8 for L/XL, 4 for B). Set it to sweep alignment depth.")
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False,
                        help="Apply LayerNorm to queries and keys inside every attention block (as in REPA's "
                             "--qk-norm). Adds q_norm/k_norm parameters, so a checkpoint trained with this flag "
                             "can only be loaded with it set the same way.")
    parser.add_argument("--num-semantic-tokens", type=int, default=1,
                        help="Number of semantic tokens pooled from teacher features and appended to the sequence "
                             "(a perfect square; 0 = no semantic tokens, i.e. plain REPA).")
    parser.add_argument("--sem-inject-max-t", type=float, default=0.5,
                        help="Only inject semantic tokens for timesteps t <= this value; for higher (noisier) t, "
                             "patch tokens are masked from attending to them. 1.0 = always inject (no masking).")
    parser.add_argument("--sem-proj-type", type=str, default="zeropad", choices=["linear", "zeropad"],
                        help="Head mapping the pooled feature (z_dim) to the SiT width. 'linear' is a "
                             "zero-initialized Linear; 'zeropad' (default) has no parameters and right-pads the pooled "
                             "feature with zeros, so the injected token IS the pooled feature. Note that with "
                             "'zeropad' semantic tokens are non-zero from step 0 (no warm-up from zero).")
    parser.add_argument("--class-dropout-prob", type=float, default=0.1,
                        help="Probability of training with the null class instead of the label, for classifier-free "
                             "guidance. 0 trains without a null class: no CFG, but autoguidance works.")

    # eval
    parser.add_argument("--eval-every", type=int, default=100000,
                        help="Run periodic eval every N steps (0 to disable)")
    parser.add_argument("--eval-samples", type=int, default=50_000)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--sampler", type=str, default="sde", choices=["ode", "sde"])

    # sample logging (wandb)
    parser.add_argument("--sampling-steps", type=int, default=3000,
                        help="Log samples to wandb every N steps (0 to disable).")
    parser.add_argument("--n-samples", type=int, default=16,
                        help="Total number of images to generate per sample-logging step.")
    parser.add_argument("--sample-cfg-scale", type=float, default=1.0,
                        help="CFG scale used for sample logging (independent from --cfg-scale used in eval).")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.class_dropout_prob == 0 and (args.cfg_scale > 1 or args.sample_cfg_scale > 1):
        parser.error("--cfg-scale and --sample-cfg-scale need a null class (--class-dropout-prob > 0)")

    # Store the resolved --encoder-depth in args, so that c2i_eval.py reads it from the checkpoint
    _size_defaults = MODEL_SIZES[args.model_size] if args.model_size is not None else {}
    if args.encoder_depth is None:
        args.encoder_depth = _size_defaults.get("encoder_depth", SiTConfig.encoder_depth)
    _depth = _size_defaults.get("depth", SiTConfig.depth)
    if not (1 <= args.encoder_depth <= _depth):
        parser.error(f"--encoder-depth must be in [1, {_depth}] for model size "
                     f"{args.model_size}; got {args.encoder_depth}. At depth > {_depth} the "
                     "`if (i + 1) == encoder_depth` branch never fires and `zs` is unbound.")

    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
