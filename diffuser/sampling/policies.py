from collections import namedtuple
import torch
import einops
import pdb

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn


Trajectories = namedtuple('Trajectories', 'actions observations values')


class GuidedPolicy:

    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1, verbose=True):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion process
        samples = self.diffusion_model(conditions, guide=self.guide, verbose=verbose, **self.sample_kwargs)

        trajectories = utils.to_np(samples.trajectories)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = trajectories[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        ## extract first action
        action = actions[0, 0]

        normed_observations = trajectories[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = Trajectories(actions, observations, samples.values)
        return action, trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device=self.device)
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions
    
    # ========================= for reverse =========================  only for special diffusion model
    def init_diffusion(self, conditions, batch_size=1, verbose=True):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion process
        self.diffusion_model(conditions, guide=self.guide, verbose=verbose, **self.sample_kwargs)

    def reverse_diffusion(self,  conditions, verbose=True):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, 1)

        ## run reverse diffusion process
        done = self.diffusion_model.reverse_sample(conditions, verbose=verbose, **self.sample_kwargs)
    
        sample = self.diffusion_model.output_sample(conditions)

        return sample, done
    
    def get_final_sample(self, conditions):
        samples = self.diffusion_model.output_sample(conditions)
        trajectories = utils.to_np(samples.trajectories)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = trajectories[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        ## extract first action
        action = actions[0, 0]

        normed_observations = trajectories[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = Trajectories(actions, observations, samples.values)
        return action, trajectories


    # ========================= for debug store =========================
    def process_raw_trajectory(self):
        x_recon_store_torch = self.diffusion_model.x_recon_store
        x_recon_store = {}

        for key, x_recon in x_recon_store_torch.items():
            sample = utils.to_np(x_recon)
            normed_observations = sample[:, :, self.action_dim:]
            observations = self.normalizer.unnormalize(normed_observations, 'observations') 

            x_recon_store[int(utils.to_np(key[0]))] = observations

        return x_recon_store

    def get_for_and_back(self):
        x_bf_store = {}
        xt_store = {}
        for t_np in reversed(range(0, self.diffusion_model.n_timesteps)):
            t = torch.tensor([t_np], device=self.device)
            x_recon, x_t = self.diffusion_model.for_and_back(t)
            x_recon = utils.to_np(x_recon)
            normed_observations = x_recon[:, :, self.action_dim:]
            observations = self.normalizer.unnormalize(normed_observations, 'observations')
            x_bf_store[t_np] = observations

            x_t = utils.to_np(x_t)
            normed_observations = x_t[:, :, self.action_dim:]
            observations = self.normalizer.unnormalize(normed_observations, 'observations')
            xt_store[t_np] = observations


        return x_bf_store, xt_store
    
    def sample_again(self):
        sample = self.diffusion_model.p_sample_loop2()
        sample = utils.to_np(sample.trajectories)
        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')
        normed_observations = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        trajectories = Trajectories(actions, observations, sample.values)
        return trajectories

    def save_values(self):
        x_value_store_torch = self.diffusion_model.x_value_store
        x_value_store = {}

        for key, x_value in x_value_store_torch.items():
            x_value_np = utils.to_np(x_value)
            # to float number
            x_value = float(x_value_np)
            x_value_store[int(utils.to_np(key[0]))] = x_value

        return x_value_store


