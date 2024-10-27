import torch

from diffuser.models.helpers import (
    extract,
    apply_conditioning,
    apply_batch_conditioning,
    get_batch_conditioning,
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

            # get cpu of y
            if y[0].cpu().detach().numpy() < 0.8:
                a = 1 # for debugging

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
def adj_cond_guided_p_sample(
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

            # get cpu of y
            if y[0].cpu().detach().numpy() < 0.8:
                a = 1 # for debugging

        if scale_grad_by_std:
            grad = model_var * grad

        grad[t < t_stopgrad] = 0

        x = x + scale * grad

        new_cond = get_batch_conditioning(x, cond, model.action_dim) # !!!!!

        x = apply_batch_conditioning(x, cond, model.action_dim)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    cond.update(new_cond) # !!!!!

    return model_mean + model_std * noise, y