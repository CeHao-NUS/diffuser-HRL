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

import os
home_dir = os.path.expanduser("~")
base_dir = '.d4rl/datasets/'

# tamp_zero = {
#     'diffusion':{
#         'horizon': 80,
#         'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.0_n10000', 'dataset.txt'),
#     },
# }


# tamp_easy = {
#     'diffusion':{
#         'horizon': 96,
#         'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.1_n10000', 'dataset.txt'),
#     },
# }


# tamp_med = {
#     'diffusion':{
#         'horizon': 112,
#         'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.3_n10000', 'dataset.txt'),
#     },
# }


# tamp_hard = {
#     'diffusion':{
#         'horizon': 128,
#         'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.5_n10000', 'dataset.txt'),
#     },
# }


# tamp_super = {
#     'diffusion':{
#         'horizon': 128,
#         'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.5_n100000', 'dataset.txt')
#     },
# }

# tamp_78 = {
#     'diffusion':{
#         'horizon': 80,
#         'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.0_n10000_78', 'dataset.txt'),
#     },
# }

# tamp_77 = {
#     'diffusion':{
#         'horizon': 80,
#         'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.0_n10000_77', 'dataset.txt'),
#     },
# }

# tamp_fix = {
#     'diffusion':{
#         'horizon': 80,
#         'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.0_n10000_fix', 'dataset.txt'),
#     },
# }


tamp_easy_78 = {
    'diffusion':{
        'horizon': 96,
        'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.1_n10000_78', 'dataset.txt'),
    },
}

tamp_med_78 = {
    'diffusion':{
        'horizon': 112,
        'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.3_n10000_78', 'dataset.txt'),
    },
}

tamp_hard_78 = {
    'diffusion':{
        'horizon': 128,
        'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.5_n10000_78', 'dataset.txt'),
    },
}

tamp_easy_2e4 = {
    'diffusion':{
        'horizon': 96,
        'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.1_n20000_78', 'dataset.txt'),
    },
}

tamp_med_2e4 = {
    'diffusion':{
        'horizon': 112,
        'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.3_n20000_78', 'dataset.txt'),
    },
}

tamp_hard_2e4 = {
    'diffusion':{
        'horizon': 128,
        'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.5_n20000_78', 'dataset.txt'),
    },
}

tamp_easy_5e4 = {
    'diffusion':{
        'horizon': 96,
        'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.1_n50000_78', 'dataset.txt'),
    },
}

tamp_med_5e4 = {
    'diffusion':{
        'horizon': 112,
        'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.3_n50000_78', 'dataset.txt'),
    },
}

tamp_hard_5e4 = {
    'diffusion':{
        'horizon': 128,
        'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.5_n50000_78', 'dataset.txt'),
    },
}

import copy
fix8_tamp_easy_78 = copy.deepcopy(tamp_easy_78)
fix8_tamp_easy_78['train_data_dir'] = os.path.join(home_dir, base_dir, 'fix8_tamp_p0.1_n10000_78', 'dataset.txt')

fix8_tamp_med_78 = copy.deepcopy(tamp_med_78)
fix8_tamp_med_78['train_data_dir'] = os.path.join(home_dir, base_dir, 'fix8_tamp_p0.3_n10000_78', 'dataset.txt')

fix8_tamp_hard_78 = copy.deepcopy(tamp_hard_78)
fix8_tamp_hard_78['train_data_dir'] = os.path.join(home_dir, base_dir, 'fix8_tamp_p0.5_n10000_78', 'dataset.txt')   

fix8_tamp_easy_2e4 = copy.deepcopy(tamp_easy_2e4)
fix8_tamp_easy_2e4['train_data_dir'] = os.path.join(home_dir, base_dir, 'fix8_tamp_p0.1_n20000_78', 'dataset.txt')

fix8_tamp_med_2e4 = copy.deepcopy(tamp_med_2e4)
fix8_tamp_med_2e4['train_data_dir'] = os.path.join(home_dir, base_dir, 'fix8_tamp_p0.3_n20000_78', 'dataset.txt')

fix8_tamp_hard_2e4 = copy.deepcopy(tamp_hard_2e4)
fix8_tamp_hard_2e4['train_data_dir'] = os.path.join(home_dir, base_dir, 'fix8_tamp_p0.5_n20000_78', 'dataset.txt')

fix8_tamp_easy_5e4 = copy.deepcopy(tamp_easy_5e4)
fix8_tamp_easy_5e4['train_data_dir'] = os.path.join(home_dir, base_dir, 'fix8_tamp_p0.1_n50000_78', 'dataset.txt')

fix8_tamp_med_5e4 = copy.deepcopy(tamp_med_5e4)
fix8_tamp_med_5e4['train_data_dir'] = os.path.join(home_dir, base_dir, 'fix8_tamp_p0.3_n50000_78', 'dataset.txt')

fix8_tamp_hard_5e4 = copy.deepcopy(tamp_hard_5e4)
fix8_tamp_hard_5e4['train_data_dir'] = os.path.join(home_dir, base_dir, 'fix8_tamp_p0.5_n50000_78', 'dataset.txt')


# ================================ evaluate ========================

# tamp_zero_eval = {
#     'diffusion':{
#         'horizon': 80,
#         'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.0_n1000_eval', 'dataset.txt'),
#     },
# }

# tamp_easy_eval = {
#     'diffusion':{
#         'horizon': 96,
#         'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.1_n1000_eval', 'dataset.txt')
#     },
# }

# tamp_med_eval = {
#     'diffusion':{
#         'horizon': 112,
#         'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.3_n1000_eval', 'dataset.txt')
#     },
# }


# tamp_hard_eval = {
#     'diffusion':{
#         'horizon': 128,
#         'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.5_n1000_eval', 'dataset.txt')
#     },
# }

# tamp_78_eval = {
#     'diffusion':{
#         'horizon': 80,
#         'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.0_n1000_78_eval', 'dataset.txt'),
#     },
# }

# tamp_77_eval = {
#     'diffusion':{
#         'horizon': 80,
#         'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.0_n1000_77_eval', 'dataset.txt'),
#     },
# }

# tamp_fix_eval = {
#     'diffusion':{
#         'horizon': 80,
#         'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.0_n1000_fix_eval', 'dataset.txt'),
#     },
# }


tamp_easy_78_eval = {
    'diffusion':{
        'horizon': 96,
        'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.0_n1000_78_eval', 'dataset.txt'),
    },
}

tamp_med_78_eval = {
    'diffusion':{
        'horizon': 112,
        'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.0_n1000_78_eval', 'dataset.txt'),
    },
}

tamp_hard_78_eval = {
    'diffusion':{
        'horizon': 128,
        'train_data_dir': os.path.join(home_dir, base_dir, 'tamp_p0.0_n1000_78_eval', 'dataset.txt'),
    },
}


fix8_tamp_easy_78_eval = copy.deepcopy(tamp_easy_78_eval)
fix8_tamp_easy_78_eval['train_data_dir'] = os.path.join(home_dir, base_dir, 'fix8_tamp_p0.0_n1000_78_eval', 'dataset.txt')

fix8_tamp_med_78_eval = copy.deepcopy(tamp_med_78_eval)
fix8_tamp_med_78_eval['train_data_dir'] = os.path.join(home_dir, base_dir, 'fix8_tamp_p0.0_n1000_78_eval', 'dataset.txt')

fix8_tamp_hard_78_eval = copy.deepcopy(tamp_hard_78_eval)
fix8_tamp_hard_78_eval['train_data_dir'] = os.path.join(home_dir, base_dir, 'fix8_tamp_p0.0_n1000_78_eval', 'dataset.txt')

##
tamp_easy_2e4_eval = tamp_easy_78_eval
tamp_med_2e4_eval = tamp_med_78_eval
tamp_hard_2e4_eval = tamp_hard_78_eval

fix8_tamp_easy_2e4_eval = fix8_tamp_easy_78_eval
fix8_tamp_med_2e4_eval = fix8_tamp_med_78_eval
fix8_tamp_hard_2e4_eval = fix8_tamp_hard_78_eval

##
tamp_easy_5e4_eval = tamp_easy_78_eval
tamp_med_5e4_eval = tamp_med_78_eval
tamp_hard_5e4_eval = tamp_hard_78_eval

fix8_tamp_easy_5e4_eval = fix8_tamp_easy_78_eval
fix8_tamp_med_5e4_eval = fix8_tamp_med_78_eval
fix8_tamp_hard_5e4_eval = fix8_tamp_hard_78_eval

