"""Train a text-to-image model: MMDiT on MS-COCO (--model mmdit) or DDT on BLIP3o (--model ddt).

MMDiT is based on REPA's train_t2i.py, and DDT on the text-to-image training of DiffusionBench.
REPA: https://github.com/sihyun-yu/REPA
DiffusionBench: https://github.com/End2End-Diffusion/diffusion-bench
"""

import argparse
import contextlib
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
import json

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from recon.data.blip3o import blip3o_loader
from recon.data.mscoco import MSCOCO256Features, features_vae_name
from recon.diffusion.loss import SILoss
from recon.diffusion.samplers import euler_sampler
from recon.diffusion.samplers_ddt import euler_sampler as ddt_euler_sampler
from recon.encoders.vision_encoder import load_encoders
from recon.eval.t2i import append_eval_csv, evaluate, evaluate_ddt
from recon.models.ddt import DDT
from recon.models.mmdit import MMDiT
from recon.models.text_encoder import Qwen3TextEncoder
from recon.models.vae import HF_VAE_DS_FACTOR, load_vae
from recon.utils import array2grid, create_logger, sample_posterior, update_ema
import wandb

logger = get_logger(__name__)

# Defaults of --model ddt: DiffusionBench's DDT recipe (configs/stage2/training/t2i/
# t2i-ddt-en28d1152hd72-dn2d2048hd128-sdvae-mse-vpred-t4.yaml) with a dinov2-vit-b teacher,
# proj_coeff 1.5, one semantic token and sem_inject_max_t 0.5
DDT_DEFAULTS = dict(
    vae_name="sdvae-ft-mse-f8d4", enc_type="dinov2-vit-b",
    proj_coeff=1.5, num_semantic_tokens=1, sem_proj_type="linear", sem_inject_max_t=0.5, num_workers=4,
    mixed_precision="bf16", allow_tf32=True, compile=True,
    batch_size=1024, max_train_steps=100_000, checkpointing_steps=10_000, sampling_steps=10_000,
    learning_rate=2e-4, adam_beta2=0.95, ema_decay=0.9995,
    lr_decay_start=50_000, lr_decay_end=1_000_000, final_learning_rate=2e-5,
    time_dist="logit-normal", t_eps=0.05, sample_cfg_scale=6.0,
    eval_every=10_000, eval_samples=10_000, eval_batch_size=64,
)


def find_resume_step(checkpoint_dir):
    """Newest step checkpoint (files named {step:07d}.pt), 0 if none."""
    if not os.path.isdir(checkpoint_dir):
        return 0
    steps = [int(f[:-3]) for f in os.listdir(checkpoint_dir)
             if f.endswith(".pt") and f[:-3].isdigit()]
    return max(steps) if steps else 0


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=accelerator_project_config,
    )

    save_dir = os.path.join(args.output_dir, args.exp_name)
    checkpoint_dir = f"{save_dir}/checkpoints"
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        os.makedirs(checkpoint_dir, exist_ok=True)  # Stores saved model checkpoints
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create model: the VAE's latent grid, in patches of 16 pixels (16x16 tokens at 256 px)
    vae, latents_scale, latents_bias = load_vae(args.vae_name, device)
    ds_factor = HF_VAE_DS_FACTOR[args.vae_name]
    assert args.resolution % 16 == 0, "Image size must be divisible by 16."
    latent_size = args.resolution // ds_factor
    patch_size, in_channels = 16 // ds_factor, vae.config.latent_channels

    # Setup REPA target encoders (repo-standard loader; dinov2-vit-b by default, "none" for no REPA)
    encoders = [] if args.enc_type == "none" else load_encoders(args.enc_type, device, args.resolution, accelerator)
    z_dims = [encoder.embed_dim for encoder in encoders]
    if args.model == "mmdit":
        model = MMDiT(
            input_size=latent_size,
            patch_size=patch_size,
            in_channels=in_channels,
            z_dims=z_dims,
            encoder_depth=args.encoder_depth,
            num_semantic_tokens=args.num_semantic_tokens,
            sem_inject_max_t=args.sem_inject_max_t,
            sem_proj_type=args.sem_proj_type,
        )
    else:
        text_encoder = Qwen3TextEncoder(device)
        null_context, null_mask = text_encoder([""])  # the empty caption, for classifier-free guidance
        model = DDT(
            input_size=latent_size,
            patch_size=patch_size,
            in_channels=in_channels,
            z_dims=z_dims,
            context_dim=text_encoder.dim,
            encoder_depth=args.encoder_depth,
            num_semantic_tokens=args.num_semantic_tokens,
            sem_inject_max_t=args.sem_inject_max_t,
        )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    ema.requires_grad_(False)

    # create loss function
    loss_fn = SILoss(time_dist=args.time_dist, t_eps=args.t_eps)
    if accelerator.is_main_process:
        logger.info(f"{type(model).__name__} Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer:
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
        fused=True if args.model == "ddt" else None,  # DDT: fused, as in DiffusionBench
    )
    # constant --learning-rate until --lr-decay-start, then linear decay to --final-learning-rate
    # at --lr-decay-end
    scheduler = None
    if args.lr_decay_end > 0:
        final_ratio = args.final_learning_rate / args.learning_rate

        def lr_lambda(step):
            if step < args.lr_decay_start:
                return 1.0
            if step >= args.lr_decay_end:
                return final_ratio
            progress = (step - args.lr_decay_start) / (args.lr_decay_end - args.lr_decay_start)
            return 1.0 - (1.0 - final_ratio) * progress

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Setup data:
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    if args.model == "mmdit":
        train_dataset = MSCOCO256Features(path=args.data_dir, cfg=True, p_uncond=args.cfg_prob).train
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=local_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        eval_dataset = MSCOCO256Features(path=args.data_dir, mode="val", ret_caption=True)
        if accelerator.is_main_process:
            logger.info(f"Loaded MS-COCO features from {args.data_dir}: {len(train_dataset):,} train images "
                        f"(captions dropped with p={args.cfg_prob}), {len(eval_dataset.test):,} val images")
    else:
        from datasets import load_from_disk
        eval_prompts = load_from_disk(args.eval_prompts)["text"][:args.eval_samples]
    eval_csv_path = os.path.join(save_dir, "evals_t2i.csv")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # resume: explicit --resume-step, else newest checkpoint in the experiment dir
    global_step = 0
    resume_step = args.resume_step or find_resume_step(checkpoint_dir)
    if resume_step > 0:
        ckpt_name = str(resume_step).zfill(7) + '.pt'
        ckpt = torch.load(
            f'{checkpoint_dir}/{ckpt_name}',
            map_location='cpu', weights_only=False,
            )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler'])
        global_step = ckpt['steps']
        if accelerator.is_main_process:
            logger.info(f"Resumed from {checkpoint_dir}/{ckpt_name} (step {global_step})")

    if args.model == "mmdit":
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )
    else:  # the BLIP3o WebDataset loader splits the shards across ranks itself
        model, optimizer = accelerator.prepare(model, optimizer)
    if args.compile:
        loss_fn = torch.compile(loss_fn)

    if accelerator.is_main_process:
        if args.report_to == "wandb" and "WANDB_KEY" in os.environ:
            wandb.login(key=os.environ["WANDB_KEY"])
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name=os.environ.get("PROJECT", "recon-t2i"),
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}", "entity": os.environ.get("ENTITY")}
            },
        )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Fixed conditioning for periodic sample logging:
    sample_batch_size = min(64 // accelerator.num_processes, local_batch_size)
    if args.model == "mmdit":
        _, gt_xs, _, _ = next(iter(train_dataloader))
        gt_xs = gt_xs[:sample_batch_size].squeeze(dim=1)
        gt_xs = sample_posterior(
            gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
            )
    # Create sampling noise:
    xT = torch.randn((sample_batch_size, in_channels, latent_size, latent_size), device=device)
    ys = gt_images = None  # conditions (and, for DDT, images) of the first batch, set in the loop
    grad_norm = torch.tensor(0.0, device=device)

    def run_eval():
        """Sample with the EMA model (MMDiT: MS-COCO val captions, ODE with Heun; DDT: the frozen BLIP3o
        prompts, Euler ODE) and append FID and CLIP similarity to evals_t2i.csv."""
        accelerator.wait_for_everyone()
        if args.model == "mmdit":
            fid, clip_sim = evaluate(
                ema, vae, latents_scale, latents_bias, accelerator,
                eval_dataset.test, eval_dataset.empty_context, args.ref_stats,
                num_samples=args.eval_samples, batch_size=args.eval_batch_size,
                num_steps=args.num_sampling_steps, cfg_scale=args.cfg_scale,
            )
        else:
            fid, clip_sim = evaluate_ddt(
                ema, vae, latents_scale, latents_bias, accelerator, text_encoder, eval_prompts, args.ref_stats,
                shard_dir=save_dir, batch_size=args.eval_batch_size,
                num_steps=args.num_sampling_steps, cfg_scale=args.cfg_scale,
            )
        if accelerator.is_main_process:
            append_eval_csv(eval_csv_path, global_step, fid, clip_sim, ckpt_key="ema", mode="ode",
                            heun=args.model == "mmdit", num_steps=args.num_sampling_steps, cfg_scale=args.cfg_scale,
                            num_samples=args.eval_samples)
            accelerator.log({"fid": fid, "clip_sim": clip_sim}, step=global_step)
            logger.info(f"Eval (step {global_step}): FID {fid:.4f}, CLIP-sim {clip_sim:.4f} -> {eval_csv_path}")

    def mmdit_batch(batch):
        """Latents sampled from the stored posteriors, the CLIP text tokens and the teacher features."""
        nonlocal ys
        raw_image, x, context, raw_captions = batch
        if ys is None:
            ys = context[:sample_batch_size].to(device)
        raw_image = raw_image.to(device)
        x = x.squeeze(dim=1).to(device)
        context = context.to(device)
        with torch.no_grad():
            x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
            zs = []
            with accelerator.autocast():
                for encoder in encoders:
                    raw_image_ = encoder.preprocess(raw_image)
                    z = encoder.forward_features(raw_image_)["x_norm_patchtokens"]
                    zs.append(z)
        return x, dict(context=context, teacher_zs=zs), zs

    def ddt_batch(batch):
        """SD-VAE latents (posterior mode), teacher features and Qwen3 text features of the images and
        captions, computed outside autocast as in DiffusionBench, with captions dropped with
        probability --cfg-prob for classifier-free guidance."""
        nonlocal ys, gt_images
        images, captions = batch
        images = images.to(device)
        with torch.no_grad():
            x = (vae.encode(images * 2 - 1).latent_dist.mode() - latents_bias) * latents_scale
            zs = [enc.forward_features(enc.preprocess(images * 255))["x_norm_patchtokens"] for enc in encoders]
            context, attn_mask = text_encoder(captions)
            if ys is None:
                ys = {"context": context[:sample_batch_size], "attn_mask": attn_mask[:sample_batch_size]}
                gt_images = images[:sample_batch_size]
            drop = torch.rand(images.shape[0], device=device) < args.cfg_prob
            context = torch.where(drop[:, None, None], null_context, context)
            attn_mask = torch.where(drop[:, None], null_mask, attn_mask)
        return x, dict(context=context, attn_mask=attn_mask, teacher_zs=zs), zs

    prepare_batch = mmdit_batch if args.model == "mmdit" else ddt_batch
    # DDT autocasts the whole loss, MMDiT only the model forward
    loss_autocast = accelerator.autocast if args.model == "ddt" else contextlib.nullcontext
    # DDT data comes in virtual epochs of --virtual-epoch-steps batches per rank
    start_epoch = (global_step * args.gradient_accumulation_steps // args.virtual_epoch_steps
                   if args.model == "ddt" else 0)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.model == "ddt":  # reshuffle the shards with --data-seed + epoch
            train_dataloader = blip3o_loader(
                args.data_dir, args.blip3o_splits.split(","), local_batch_size, args.num_workers,
                epoch, args.virtual_epoch_steps, seed=args.data_seed, shuffle_buffer=args.shuffle_buffer,
            )
        for batch in train_dataloader:
            x, model_kwargs, zs = prepare_batch(batch)

            with accelerator.accumulate(model):
                with loss_autocast():
                    loss, proj_loss = loss_fn(model, x, model_kwargs, zs=zs)
                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean()
                loss = loss_mean + proj_loss_mean * args.proj_coeff

                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model, decay=args.ema_decay)
                    if scheduler is not None:
                        scheduler.step()

            if not accelerator.sync_gradients:
                continue  # still accumulating: no optimizer step was taken
            progress_bar.update(1)
            global_step += 1
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    if scheduler is not None:
                        checkpoint["scheduler"] = scheduler.state_dict()
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if args.eval_every > 0 and global_step % args.eval_every == 0:
                run_eval()

            if args.sampling_steps > 0 and (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                with torch.no_grad():
                    if args.model == "mmdit":
                        y_null = torch.tensor(train_dataset.empty_token).to(device).unsqueeze(0).repeat(ys.shape[0], 1, 1)
                        samples = euler_sampler(
                            ema,
                            xT,
                            {"context": ys},
                            {"context": y_null},
                            num_steps=50,
                            cfg_scale=args.sample_cfg_scale,
                            guidance_low=0.,
                            guidance_high=1.,
                            heun=False,
                        ).to(torch.float32)
                        samples = vae.decode(samples / latents_scale + latents_bias).sample
                        gt_samples = vae.decode(gt_xs / latents_scale + latents_bias).sample
                        samples = (samples + 1) / 2.
                        gt_samples = (gt_samples + 1) / 2.
                    else:
                        y_null = {"context": null_context.expand_as(ys["context"]),
                                  "attn_mask": null_mask.expand_as(ys["attn_mask"])}
                        with accelerator.autocast():
                            samples = ddt_euler_sampler(ema, xT, ys, y_null, num_steps=50,
                                                        cfg_scale=args.sample_cfg_scale)
                        samples = (vae.decode(samples / latents_scale + latents_bias).sample + 1) / 2.
                        gt_samples = gt_images
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                if accelerator.is_main_process and args.report_to == "wandb":
                    accelerator.log({"samples": wandb.Image(array2grid(out_samples)),
                                     "gt_samples": wandb.Image(array2grid(gt_samples))})
                logging.info("Generating EMA samples done.")

            if global_step % args.log_every == 0:
                logs = {
                    "loss": accelerator.gather(loss_mean).mean().detach().item(),
                    "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                    "grad_norm": accelerator.gather(grad_norm).mean().detach().item(),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                if accelerator.is_main_process:
                    logger.info(
                        f"step={global_step}  epoch={epoch}  loss={logs['loss']:.4f}  "
                        f"proj_loss={logs['proj_loss']:.4f}  grad_norm={logs['grad_norm']:.4f}"
                    )

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    # Final eval, unless the loop just ran it
    if not (args.eval_every > 0 and global_step % args.eval_every == 0):
        run_eval()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--model", type=str, default="mmdit", choices=["mmdit", "ddt"],
                        help="mmdit: MMDiT on MS-COCO (REPA's t2i setup); ddt: DDT on BLIP3o, with "
                             "DiffusionBench's recipe as the defaults (DDT_DEFAULTS)")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb",
                        help="Tracker backend for accelerate (e.g. 'wandb', 'tensorboard', or 'none').")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--sample-cfg-scale", type=float, default=4.0,
                        help="CFG scale used for sample logging (independent from --cfg-scale used in eval).")
    parser.add_argument("--resume-step", type=int, default=0,
                        help="0 = auto-resume from newest checkpoint if present")
    parser.add_argument("--log-every", type=int, default=100)

    # model
    parser.add_argument("--encoder-depth", type=int, default=8)

    # dataset
    parser.add_argument("--data-dir", type=str, required=True,
                        help="MMDiT: MS-COCO features (preprocessing/extract_mscoco_features.py); "
                             "DDT: BLIP3o-256 shards, one directory per split")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--vae-name", type=str, default="sdvae-ft-ema-f8d4",
                        help="VAE (recon/models/vae.py). MMDiT: the VAE the features were extracted with "
                             "(or the other SD-VAE, which shares its encoder); DDT encodes the images with it")
    parser.add_argument("--blip3o-splits", type=str, default="journeydb,long-caption,short-caption",
                        help="DDT: comma-separated BLIP3o splits (subdirectories of --data-dir)")
    parser.add_argument("--virtual-epoch-steps", type=int, default=10000,
                        help="DDT: batches per rank in each pass over the BLIP3o stream (optimizer steps times "
                             "--gradient-accumulation-steps); each pass reshuffles the shards")
    parser.add_argument("--data-seed", type=int, default=42, help="DDT: seed of the shard and sample shuffling")
    parser.add_argument("--shuffle-buffer", type=int, default=10000, help="DDT: sample shuffle buffer per worker")

    # precision
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                        help="torch.compile the loss (model forward included)")

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=150000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-decay-start", type=int, default=0,
                        help="Step at which a linear learning-rate decay starts (with --lr-decay-end > 0)")
    parser.add_argument("--lr-decay-end", type=int, default=0,
                        help="Step at which the decay reaches --final-learning-rate; 0 = constant learning rate")
    parser.add_argument("--final-learning-rate", type=float, default=2e-5)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=8)

    # loss
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b',
                        help="Comma-separated teacher encoders; 'none' trains without REPA")
    parser.add_argument("--proj-coeff", type=float, default=1.5)
    parser.add_argument("--time-dist", type=str, default="uniform", choices=["uniform", "logit-normal"],
                        help="Training distribution of t: U(0, 1) or sigmoid(N(0, 1))")
    parser.add_argument("--t-eps", type=float, default=0.0,
                        help="> 0: velocity target (x_t - x) / max(t, t_eps) instead of noise - x")

    # eval (EMA weights; training also ends with an eval)
    parser.add_argument("--eval-every", type=int, default=50000,
                        help="Run periodic eval every N steps (0 to disable)")
    parser.add_argument("--eval-samples", type=int, default=10240,
                        help="Captions to sample (MMDiT: divisible by --eval-batch-size x number of GPUs). "
                             "Use t2i_eval.py for the full protocol (40192 MS-COCO / 50000 BLIP3o samples).")
    parser.add_argument("--eval-prompts", type=str, default=None,
                        help="DDT: the prompts of the frozen reference (preprocessing/build_blip3o_reference.py)")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Per-GPU sampling batch size for eval.")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="CFG scale used in eval (1.0 = no guidance).")
    parser.add_argument("--num-sampling-steps", type=int, default=50, help="Sampling steps used in eval.")
    parser.add_argument("--ref-stats", type=str, required=True,
                        help="FID reference statistics (MMDiT: U-ViT's fid_stats_mscoco256_val.npz; "
                             "DDT: reference_stats.npz of preprocessing/build_blip3o_reference.py)")

    # semantic tokens
    parser.add_argument("--num-semantic-tokens", type=int, default=1,
                        help="Number of semantic tokens pooled from teacher features and appended to the sequence "
                             "(a perfect square; 0 = no semantic tokens, i.e. plain REPA).")
    parser.add_argument("--sem-inject-max-t", type=float, default=0.5,
                        help="Only inject semantic tokens for timesteps t <= this value; for higher (noisier) t, "
                             "tokens are masked from attending to them. 1.0 = always inject (no masking). "
                             "Training-time only; at eval semantic tokens are always injected.")
    parser.add_argument("--sem-proj-type", type=str, default="zeropad", choices=["linear", "zeropad"],
                        help="Head mapping the pooled feature (z_dim) to the model width. 'linear' is a "
                             "zero-initialized Linear; 'zeropad' (default) has no parameters and right-pads the pooled "
                             "feature with zeros (identity here since z_dim == hidden_size == 768).")

    args = parser.parse_args(input_args)
    if args.model == "ddt":
        parser.set_defaults(**DDT_DEFAULTS)
        args = parser.parse_args(input_args)
        if args.sem_proj_type != "linear":
            parser.error("--model ddt supports --sem-proj-type linear only")
        if args.eval_prompts is None:
            parser.error("--model ddt needs --eval-prompts (preprocessing/build_blip3o_reference.py)")
    if not os.path.isfile(args.ref_stats):
        parser.error(f"--ref-stats {args.ref_stats} is not a file")
    if args.model == "mmdit":
        features_vae = features_vae_name(args.data_dir)
        sd_vaes = {"sdvae-ft-ema-f8d4", "sdvae-ft-mse-f8d4"}  # one encoder, two fine-tuned decoders
        if args.vae_name != features_vae and not {args.vae_name, features_vae} <= sd_vaes:
            parser.error(f"--vae-name {args.vae_name} does not match the features in {args.data_dir}, "
                         f"extracted with {features_vae}")
    return args

if __name__ == "__main__":
    args = parse_args()

    main(args)
