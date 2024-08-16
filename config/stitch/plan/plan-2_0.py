
'''
1. load two diffusers

'''

from diffuser.utils import watch
import diffuser.sampling as sampling

plan_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('value_horizon', 'V'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ('seg_length', 'L'),
    ##
    ('conditional', 'cond'),
]



base = {
    'plan': {


        'LL_diffusion_loadpath': 'f:diffusion/LL_diffuser_H{LL_horizon}_T{n_diffusion_steps}',
        'LL_diffusion_epoch': 'latest',

        'HL_diffusion_loadpath': 'f:diffusion/HL_diffuser_H{HL_horizon}_T{n_diffusion_steps}_D{downsample}',
        'HL_diffusion_epoch': 'latest',
        'downsample': 32,

        'seg_length': 3,

        # 'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 1000,
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': None,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 1.0,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': 'logs',
        'prefix': 'plans/plan-2_0/',
        'exp_name': watch(plan_args_to_watch),
        'vis_freq': 10,
        'max_render': 8,

        ## diffusion model
        'horizon': 256,
        'n_diffusion_steps': 256,

        ## value function
        'discount': 0.99,

        ## loading
        

        'verbose': True,
        'suffix': '0',

        # setting
        'conditional': False,
        'init_pose': None,
        'target': None,
    },
}

maze2d_umaze_v1 = {
    'plan': {
        'LL_horizon': 32,
        'HL_horizon': 128,
        'n_diffusion_steps': 32,
    },
}