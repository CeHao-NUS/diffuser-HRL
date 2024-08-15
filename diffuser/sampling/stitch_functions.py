import torch

@torch.no_grad()
def default_sample_fn(model, x, cond, t, **kwargs):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values

@torch.no_grad()
def LL_LL_joint_sample_fn(model, x, cond, t, **kwargs):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    new_x =  model_mean + model_std * noise

    for i in range(len(x)-1):
        joint_value = (new_x[i,-1] + new_x[i+1,0]) / 2
        new_x[i,-1] = joint_value
        new_x[i+1,0] = joint_value

    return new_x, values