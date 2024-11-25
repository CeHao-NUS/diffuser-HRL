import socket

from diffuser.utils import watch


diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]


base = {
    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 128,
        'n_diffusion_steps': 64,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.StateTrajRenderer',

        ## dataset
        'loader': 'datasets.StateTrajBitDataset',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 40000,

        'n_bits': 6,
        'n_objs': 2,
        'cond_index': [0, 1, 2, 3, 4, 5, 6],

        'tokenizer_save_path': './custom_tokenizer',
        'train_data_dir': '', # TODO

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion_bits/single_diffuser',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 2e6,
        'batch_size': 64,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 2000,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cuda',

    },


    
}


tamp_easy = {
    'diffusion':{
        'horizon': 96,
        'train_data_dir': "/home/crtie/.d4rl/datasets/tamp_p0.1_n10000/dataset.txt", 
    },
}

