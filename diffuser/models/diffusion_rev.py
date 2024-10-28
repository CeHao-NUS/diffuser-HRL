from diffuser.models.stitch_diffsuion import *



SampleRev = namedtuple('Sample', 'trajectories values chains x_recon')

class GaussianDiffusionnRev(BatchGaussianDiffusion):

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_batch_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()

        self.sample_dict = {
            'device': device,
            'batch_size': batch_size,
            'x': x,
            'chain': chain,
            'progress': progress,
            'values': None,
            'i': self.n_timesteps-1,
        }
        
    
    @torch.no_grad()
    def reverse_sample(self, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        # for i in reversed(range(0, self.n_timesteps)):

        device = self.sample_dict['device']
        batch_size = self.sample_dict['batch_size']
        x = self.sample_dict['x']
        chain = self.sample_dict['chain']
        progress = self.sample_dict['progress']
        i = self.sample_dict['i']

        t = make_timesteps(batch_size, i, device)

        x, values = sample_fn(self, x, cond, t, **sample_kwargs)
        x = apply_batch_conditioning(x, cond, self.action_dim)

        progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
        if return_chain: chain.append(x)

        self.sample_dict['x'] = x
        self.sample_dict['chain'] = chain
        self.sample_dict['values'] = values
        self.sample_dict['i'] = i - 1

        return self.sample_dict['i']


    @torch.no_grad()
    def output_sample(self, cond, return_chain=False):
        
        x = self.sample_dict['x']
        values = self.sample_dict['values']
        chain = self.sample_dict['chain']
        progress = self.sample_dict['progress']
        i = self.sample_dict['i']
        batch_size = self.sample_dict['batch_size']
        device = self.sample_dict['device']

        t = make_timesteps(batch_size, i, device)

        x_recon = self.p_mean_variance(x, cond, t)
                                    
        progress.stamp()

        x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain, x_recon)