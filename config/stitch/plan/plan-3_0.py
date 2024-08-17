import socket

from diffuser.utils import watch

plan_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('value_horizon', 'V'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ##
    ('conditional', 'cond'),
    ('min_horizon', 'mH'),
]

base = {
    'plan': {
        ## loading
        'diffusion_loadpath': 'f:diffusion/LL_varh_diffuser_H{horizon}_T{n_diffusion_steps}_mH{min_horizon}',
        'value_loadpath': 'f:values/LL_varh_value_H{horizon}_T{n_diffusion_steps}_d{discount}_mH{min_horizon}',

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',


        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 1000,
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': None,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.5,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': 'logs',
        'prefix': 'plans/guided_varh',
        'exp_name': watch(plan_args_to_watch),
        'vis_freq': 10,
        'max_render': 8,

        ## diffusion model
        'horizon': 256,
        'n_diffusion_steps': 256,
        'min_horizon': 16,

        ## value function
        'discount': 0.99,

        'verbose': True,
        'suffix': '0',

        # setting
        'conditional': False,
        'init_pose': None,
        'target': None,
    },
}

maze2d_umaze_v1 = {
    'plan':{
        'horizon': 128,
        'n_diffusion_steps': 64,
        'min_horizon': 16,
    },

}

maze2d_medium_v1 = {
    'plan':{
        'horizon': 256,
        'n_diffusion_steps': 256,
        'min_horizon': 16,
    },

}

maze2d_large_v1 = {
    'plan':{
        'horizon': 384,
        'n_diffusion_steps': 256,
        'min_horizon': 48,
    },

}
