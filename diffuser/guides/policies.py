from collections import namedtuple
# import numpy as np
import torch
import einops
import pdb

import diffuser.utils as utils
# from diffusion.datasets.preprocessing import get_policy_preprocess_fn

Trajectories = namedtuple('Trajectories', 'actions observations')
# GuidedTrajectories = namedtuple('GuidedTrajectories', 'actions observations value')

class Policy:

    def __init__(self, diffusion_model, normalizer, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim
        self.sample_kwargs = sample_kwargs

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

    def __call__(self, conditions, debug=False, batch_size=1):


        conditions = self._format_conditions(conditions, batch_size)

        ## batchify and move to tensor [ batch_size x observation_dim ]
        # observation_np = observation_np[None].repeat(batch_size, axis=0)
        # observation = utils.to_torch(observation_np, device=self.device)

        ## run reverse diffusion process
        sample = self.diffusion_model(conditions, **self.sample_kwargs)
        sample = utils.to_np(sample.trajectories)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')
        # actions = np.tanh(actions)

        ## extract first action
        action = actions[0, 0]

        # if debug:
        normed_observations = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        # if deltas.shape[-1] < observation.shape[-1]:
        #     qvel_dim = observation.shape[-1] - deltas.shape[-1]
        #     padding = np.zeros([*deltas.shape[:-1], qvel_dim])
        #     deltas = np.concatenate([deltas, padding], axis=-1)

        # ## [ batch_size x horizon x observation_dim ]
        # next_observations = observation_np + deltas.cumsum(axis=1)
        # ## [ batch_size x (horizon + 1) x observation_dim ]
        # observations = np.concatenate([observation_np[:,None], next_observations], axis=1)

        trajectories = Trajectories(actions, observations)
        return action, trajectories
        # else:
        #     return action

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
        trajectories = Trajectories(actions, observations)
        return trajectories