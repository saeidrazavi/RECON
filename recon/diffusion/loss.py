import torch
import torch.nn.functional as F

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

class SILoss:
    """Velocity (v-prediction) loss on the linear interpolant x_t = (1 - t) * x + t * noise, plus the
    REPA projection loss. t ~ U(0, 1), or t ~ sigmoid(N(0, 1)) with time_dist="logit-normal". With
    t_eps > 0 the target is (x_t - x) / max(t, t_eps): the velocity noise - x for t >= t_eps,
    scaled down below it (the DDT recipe of DiffusionBench)."""

    def __init__(self, time_dist="uniform", t_eps=0.0):
        assert time_dist in ("uniform", "logit-normal"), time_dist
        self.time_dist = time_dist
        self.t_eps = t_eps

    def __call__(self, model, images, model_kwargs=None, zs=None):
        if model_kwargs is None:
            model_kwargs = {}
        if self.time_dist == "logit-normal":
            time_input = torch.randn((images.shape[0], 1, 1, 1)).sigmoid()
        else:
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        time_input = time_input.to(device=images.device, dtype=images.dtype)

        noises = torch.randn_like(images)
        model_input = (1 - time_input) * images + time_input * noises
        if self.t_eps > 0:
            model_target = (model_input - images) / time_input.clamp_min(self.t_eps)
        else:
            model_target = noises - images  # velocity dx_t/dt
        model_output, zs_tilde  = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # projection (REPA) loss: negative cosine similarity between the projected
        # model features and the teacher features, averaged over tokens, samples
        # and teachers.
        proj_loss = 0.
        for z, z_tilde in zip(zs, zs_tilde):
            proj_loss += -(F.normalize(z, dim=-1) * F.normalize(z_tilde, dim=-1)).sum(dim=-1).mean()
        proj_loss = proj_loss / len(zs) if zs else torch.zeros((), device=images.device)  # no teacher

        return denoising_loss, proj_loss
