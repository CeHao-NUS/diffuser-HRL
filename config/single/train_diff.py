import socket

from diffuser.utils import watch


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

base = {
    # d-1: LL
    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 320,
        'n_diffusion_steps': 256,
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
        'prefix': 'diffusion/single_diffuser',
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

    'values': {
        'model': 'models.ValueFunction',
        'diffusion': 'models.ValueDiffusion',
        'horizon': 320,
        'n_diffusion_steps': 256,
        'dim_mults': (1, 2, 2, 2, 4, 4, 8),
        'renderer': 'utils.Maze2dRenderer',

        ## value-specific kwargs
        'discount': 0.99,
        'termination_penalty': None,
        'normed': False,

        ## dataset
        'loader': 'datasets.AverageValueDataset',
        'normalizer': 'LimitsNormalizer',
        # 'preprocess_fns': ['maze2d_set_terminals'],
        'preprocess_fns': [],
        'use_padding': False,
        'max_path_length': 40000,
        'min_horizon': 16,
        'h5path': 'temp_datasets/integrated.h5', # !!!!
        # 'h5path': 'temp_datasets/med_single_maze2d.hdf5', # !!!!

        ## serialization
        'logbase': 'logs',
        'prefix': 'values/single_diffuser',
        'exp_name': watch(value_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'value_l2',
        'n_train_steps': 400e3,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 0,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
    },
}

# '''
# original diffuser
maze2d_umaze_v1 = {
    'diffusion':{
        'horizon': 128,
        'n_diffusion_steps': 64,
    },

    'values':{
        'horizon': 128,
        'n_diffusion_steps': 64,
    },
}

maze2d_medium_v1 = {
    'diffusion':{
        'horizon': 32,
        'n_diffusion_steps': 320,
    },

    'values':{
        'horizon': 320,
        'n_diffusion_steps': 320,
    },
}

maze2d_large_v1 = {
    'diffusion':{
        'horizon': 448,
        'n_diffusion_steps': 256,
    },

    'values':{
        'horizon': 448,
        'n_diffusion_steps': 256,
    },
}
