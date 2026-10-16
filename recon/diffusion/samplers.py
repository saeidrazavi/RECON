"""Euler ODE (optionally Heun) and Euler-Maruyama SDE samplers for the linear interpolant
x_t = (1 - t) * x + t * noise, integrated from t = 1 (noise) to t = 0 (data).

`cond` and `null_cond` are dicts of the model's condition keyword arguments, e.g.
{"y": labels} and {"y": torch.full_like(labels, 1000)} for SiT, or {"context": tokens}
and {"context": empty_tokens} for MMDiT. Classifier-free guidance with scale cfg_scale
is applied while guidance_low <= t <= guidance_high. Autoguidance (weak_model, ag_scale) uses the same interval;
ag_projected and ste_lambda select projected autoguidance and Semantic Token Extrapolation (SiT only).
"""
import torch


def expand_t_like_x(t, x_cur):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x_cur.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

def get_score_from_velocity(vt, xt, t):
    """Score of the linear interpolant x_t = (1 - t) * x + t * noise, from the velocity prediction.
    Args:
        vt: [batch_dim, ...] shaped tensor; velocity model output
        xt: [batch_dim, ...] shaped tensor; x_t data point
        t: [batch_dim,] time tensor
    """
    t = expand_t_like_x(t, xt)
    alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
    sigma_t, d_sigma_t = t, torch.ones_like(xt, device=xt.device)

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var

    return score


def compute_diffusion(t_cur):
    return 2 * t_cur


def guided_velocity(model, x, t, cond, null_cond, cfg_scale, use_cfg, dtype):
    """Velocity predicted by `model` at time t (float64 in, float64 out). With use_cfg,
    the unconditional branch (null_cond) runs in the same batch and classifier-free
    guidance is applied."""
    if use_cfg:
        x = torch.cat([x, x], dim=0)
        cond = {k: torch.cat([v, null_cond[k]], dim=0) for k, v in cond.items()}
    time_input = torch.ones(x.size(0), device=x.device, dtype=torch.float64) * t
    v = model(x.to(dtype=dtype), time_input.to(dtype=dtype), **cond)[0].to(torch.float64)
    if use_cfg:
        v_cond, v_uncond = v.chunk(2)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
    return v


class SemanticTokenExtrapolation:
    """Moves the unit pooled token u away from the token u_1 predicted at the first step:
    u' = u + lam * (u - u_1). One instance per sampling run."""
    def __init__(self, lam):
        self.lam = lam
        self.anchor = None

    def __call__(self, u):
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.anchor is None:
            self.anchor = u
        return u + self.lam * (u - self.anchor)


def autoguided_velocity(model, weak_model, x, t, cond, ag_scale, ag_projected, ste, dtype):
    """Velocity of `model` (with the Semantic Token Extrapolation hook `ste`), guided by `weak_model` when
    ag_scale is not None (float64 in, float64 out). Projected autoguidance removes, at each latent position,
    the part of the guidance update that is parallel to the clean estimate x0 = x - t v of `model`."""
    x = x.to(dtype=dtype)
    time_input = (torch.ones(x.size(0), device=x.device, dtype=torch.float64) * t).to(dtype=dtype)
    v = model(x, time_input, **cond, **({"sem_extrapolation": ste} if ste is not None else {}))[0]
    if ag_scale is not None:
        v_weak = weak_model(x, time_input, **cond)[0]
        if ag_projected:
            t = float(t)
            x0 = x - t * v
            dx = -t * (v - v_weak)  # guidance update in x0 space
            par = (dx * x0).sum(1, keepdim=True) / (x0 * x0).sum(1, keepdim=True).clamp_min(1e-8) * x0
            v = v + (ag_scale - 1.0) * (-(dx - par) / t)
        else:
            v = v_weak + ag_scale * (v - v_weak)
    return v.to(torch.float64)


def make_velocity_fn(model, cond, null_cond, cfg_scale, dtype, weak_model=None, ag_scale=None, ag_projected=False,
                     ste_lambda=None):
    """velocity(x, t, active): the sampler's velocity; `active` says if t is inside the guidance interval."""
    if (weak_model is None) != (ag_scale is None):
        raise ValueError("autoguidance needs both weak_model and ag_scale")
    if (ag_scale is not None or ste_lambda is not None) and cfg_scale > 1.0:
        raise ValueError("use either classifier-free guidance or autoguidance / Semantic Token Extrapolation")
    if weak_model is None and ste_lambda is None:
        return lambda x, t, active: guided_velocity(model, x, t, cond, null_cond, cfg_scale,
                                                    cfg_scale > 1.0 and active, dtype)
    ste = SemanticTokenExtrapolation(ste_lambda) if ste_lambda is not None else None
    return lambda x, t, active: autoguided_velocity(model, weak_model, x, t, cond, ag_scale if active else None,
                                                    ag_projected, ste, dtype)


def euler_sampler(
        model,
        latents,
        cond,
        null_cond,
        num_steps=20,
        heun=False,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        weak_model=None,
        ag_scale=None,
        ag_projected=False,
        ste_lambda=None,
        ):
    dtype = latents.dtype
    velocity = make_velocity_fn(model, cond, null_cond, cfg_scale, dtype, weak_model, ag_scale, ag_projected, ste_lambda)
    t_steps = torch.linspace(1, 0, num_steps + 1, dtype=torch.float64)
    x_next = latents.to(torch.float64)

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            active = guidance_low <= t_cur <= guidance_high
            d_cur = velocity(x_cur, t_cur, active)
            x_next = x_cur + (t_next - t_cur) * d_cur
            if heun and i < num_steps - 1:
                d_prime = velocity(x_next, t_next, active)
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def euler_maruyama_sampler(
        model,
        latents,
        cond,
        null_cond,
        num_steps=20,
        heun=False,  # unused, kept for a signature compatible with euler_sampler
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        weak_model=None,
        ag_scale=None,
        ag_projected=False,
        ste_lambda=None,
        ):
    dtype = latents.dtype
    velocity = make_velocity_fn(model, cond, null_cond, cfg_scale, dtype, weak_model, ag_scale, ag_projected, ste_lambda)
    t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
    x_next = latents.to(torch.float64)

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            dt = t_next - t_cur
            v_cur = velocity(x_cur, t_cur, guidance_low <= t_cur <= guidance_high)
            time_input = torch.ones(x_cur.size(0), device=x_cur.device, dtype=torch.float64) * t_cur
            s_cur = get_score_from_velocity(v_cur, x_cur, time_input)
            diffusion = compute_diffusion(t_cur)
            d_cur = v_cur - 0.5 * diffusion * s_cur
            if i < num_steps - 1:
                deps = torch.randn_like(x_cur) * torch.sqrt(torch.abs(dt))
                x_next = x_cur + d_cur * dt + torch.sqrt(diffusion) * deps
            else:
                # last step (t = 0.04 -> 0) is deterministic: return the mean
                x_next = x_cur + d_cur * dt

    return x_next
