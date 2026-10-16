"""Euler ODE sampler for DDT, as in DiffusionBench (Sampler.sample_ode with forward_with_cfg):
float32 state on a uniform time grid from t = 1 (noise) to t = 0 (data), with classifier-free
guidance of scale cfg_scale while guidance_low <= t <= guidance_high. Run it under bf16 autocast,
like DDT training.

cond and null_cond are dicts of the model's condition keyword arguments, e.g.
{"context": tokens, "attn_mask": mask} for the captions and for the empty caption.
"""
import torch


@torch.no_grad()
def euler_sampler(model, latents, cond, null_cond, num_steps=50, cfg_scale=1.0, guidance_low=0.0, guidance_high=1.0):
    use_cfg = cfg_scale > 1.0
    x = latents
    if use_cfg:  # one batch: conditional half first, unconditional half second
        x = torch.cat([x, x], dim=0)
        cond = {k: torch.cat([v, null_cond[k]], dim=0) for k, v in cond.items()}
    t_steps = torch.linspace(1.0, 0.0, num_steps + 1).to(x.device)
    for i in range(num_steps):
        h = t_steps[i] - t_steps[i + 1]
        t = torch.full((x.shape[0],), t_steps[i].item(), device=x.device)
        v = model(x, t, **cond)[0]
        if use_cfg:
            v_cond, v_uncond = v.chunk(2)
            t_half = t[:len(t) // 2]
            in_interval = ((t_half >= guidance_low) & (t_half <= guidance_high)).view(-1, 1, 1, 1)
            v = torch.where(in_interval, v_uncond + cfg_scale * (v_cond - v_uncond), v_cond)
            v = torch.cat([v, v], dim=0)
        x = x - h * v
    return x.chunk(2)[0] if use_cfg else x
