from diffuser.models.diffusion import *
from diffuser.models.helpers import *

Sample_rope = namedtuple('Sample', 'trajectories values')

class RopeDiffusion(GaussianDiffusion):

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, training=True, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        
        # batch_size = len(cond[0])

        # training mode
        if training:
            batch_size = len(cond[0])
        else:
            batches = [k[0] for k in cond.keys()]
            batch_size = np.max(batches) + 1

        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, **sample_kwargs)
    
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True,  sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_batch_conditioning(x, cond, self.action_dim)

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)
            x = apply_batch_conditioning(x, cond, self.action_dim)

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
        progress.stamp()

        return Sample_rope(x, values)
    
    # def p_losses(self, x_start, cond, t):
    #     noise = torch.randn_like(x_start)

    #     x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    #     x_noisy = apply_batch_conditioning(x_noisy, cond, self.action_dim)

    #     x_recon = self.model(x_noisy, cond, t)
    #     x_recon = apply_batch_conditioning(x_recon, cond, self.action_dim)

    #     assert noise.shape == x_recon.shape

    #     if self.predict_epsilon:
    #         loss, info = self.loss_fn(x_recon, noise)
    #     else:
    #         loss, info = self.loss_fn(x_recon, x_start)

    #     return loss, info