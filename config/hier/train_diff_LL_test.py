# LL fixed length

import socket

from diffuser.utils import watch


diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]

base = {
    # d-1: LL
    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 256,
        'n_diffusion_steps': 256,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.Maze2dRenderer',

        ## dataset
        'loader': 'datasets.LL_goal_dataset',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 40000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/LL_diffuser',
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
        'sample_freq': 5000,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cuda',

    },
}

maze2d_testbig_v0 = {
    'diffusion':{
        'horizon': 32,
        'n_diffusion_steps': 32,
    },
}


maze2d_umaze_v1 = {
    'diffusion': {
        'horizon': 16,
        'n_diffusion_steps': 32,
    },
}

maze2d_medium_v1 = {
    'diffusion': {
        'horizon': 32,
        'n_diffusion_steps': 32,
    },
}

maze2d_large_v1 = {
    'diffusion': {
        'horizon': 32,
        'n_diffusion_steps': 32,
    },
}
