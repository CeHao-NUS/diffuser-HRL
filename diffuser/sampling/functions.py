import torch

from diffuser.models.helpers import (
    extract,
    apply_conditioning,
    apply_batch_conditioning
)

@torch.no_grad()
def default_sample_fn(model, x, cond, t):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


@torch.no_grad()
def n_step_guided_p_sample(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True,
):
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    for _ in range(n_guide_steps):
        with torch.enable_grad():
            x_no_action = x.clone()
            x_no_action[:, :, :model.action_dim] = 0
            y, grad = guide.gradients(x_no_action, cond, t)

        if scale_grad_by_std:
            grad = model_var * grad

        grad[t < t_stopgrad] = 0

        x = x + scale * grad
        # x = apply_conditioning(x, cond, model.action_dim)
        x = apply_batch_conditioning(x, cond, model.action_dim)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y

@torch.no_grad()
def n_step_guided_p_sample_rope(model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True,
):
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    batch_size = x.shape[0]

    for _ in range(n_guide_steps):
        with torch.enable_grad():
            
            grad = torch.zeros_like(x)
            y = torch.zeros(batch_size-1, device=x.device)

            for idx in range(batch_size-1):
                x_prev = x[idx, -1, 2:4] # two dim for pose
                x_next = x[idx+1, 0, 2:4] # two dim for pose
                y_i, grad_i = guide.gradients(x_prev, x_next)

                grad[idx, -1, 2:4] = grad_i
                grad[idx+1, 0, 2:4] = - grad_i
                y[idx] = y_i

        if scale_grad_by_std:
            model_var = torch.clamp(model_var, min=1e-3)
            grad = model_var * grad

        
        # grad[t < t_stopgrad] = 0
        scale = 1

        x = x + scale * grad
        x = apply_batch_conditioning(x, cond, model.action_dim)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y