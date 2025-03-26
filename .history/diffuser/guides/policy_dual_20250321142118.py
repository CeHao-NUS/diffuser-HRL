from collections import namedtuple
# import numpy as np
import torch
import einops
import pdb

import diffuser.utils as utils
# from diffusion.datasets.preprocessing import get_policy_preprocess_fn

Trajectories = namedtuple('Trajectories', 'actions observations')
# GuidedTrajectories = namedtuple('GuidedTrajectories', 'actions observations value')

class PolicyDual:

    def __init__(self, diffusion_model_hl, diffusion_model_ll, normalizer_hl, normalizer_ll, **sample_kwargs):
        self.diffusion_model_hl = diffusion_model_hl
        self.diffusion_model_ll = diffusion_model_ll
        self.normalizer_hl = normalizer_hl
        self.normalizer_ll = normalizer_ll
        self.action_dim_hl = normalizer_hl.action_dim
        self.action_dim_ll = normalizer_ll.action_dim

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

    def __call__(self, conditions, batch_size=1):

           

        conditions = self._format_conditions(conditions, batch_size)

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

        trajectories = Trajectories(actions, observations)
        return action, trajectories
