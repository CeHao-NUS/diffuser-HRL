import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]

value_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ## value kwargs
    ('discount', 'd'),
]


plan_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('scale', 'Vs'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ##
    ('conditional', 'cond'),
]

base = {

    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.RopeDiffusion',
        'horizon': 64,
        'n_diffusion_steps': 32,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.Maze2dRenderer',

        ## dataset
        'loader': 'datasets.GoalDataset',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 40000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/rope',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 2e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 1000,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cuda',
    },

    
    'plan': {
        'policy': 'sampling.RopePolicy',
        'max_episode_length': 1000,
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': None,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': 'logs',
        'prefix': 'plans/rope',
        'exp_name': watch(plan_args_to_watch),
        'vis_freq': 10,
        'max_render': 8,

        ## diffusion model
        'diffusion_horizon': 64,
        'diffusion_n_diffusion_steps': 32,

        ## value function
        'value_horizon': 256,
        'value_n_diffusion_steps': 256,
        'discount': 0.995,

        ## loading
        'diffusion_loadpath': 'f:diffusion/rope_H{diffusion_horizon}_T{diffusion_n_diffusion_steps}',

        'diffusion_epoch': 'latest',

        'verbose': True,
        'suffix': '0',

        # setting
        'conditional': False,
        'init_pose': None,
        'target': None,
    },

}

#------------------------ overrides ------------------------#

'''
    maze2d maze episode steps:
        umaze: 150
        medium: 250
        large: 600
'''

maze2d_umaze_v1 = {
    'diffusion': {
        'horizon': 32,
        'n_diffusion_steps': 32,
    },

    'plan': {
        'diffusion_horizon': 32,
        'diffusion_n_diffusion_steps': 32,

    },
}

