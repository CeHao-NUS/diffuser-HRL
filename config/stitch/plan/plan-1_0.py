import socket

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
    ##
    ('conditional', 'cond'),
]


base = {
    'plan': {


        'diffusion_loadpath': 'f:diffusion/LL_diffuser_H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',


        # 'classifier': [
        #     {'guide': None, 
        #     'value_loadpath': '',
        #     'sample_fun': ''}
        # ],

        'guide_LL': None,
        'value_LL': None,
        'sample_fun': sampling.stitch_functions.default_sample_fn,

        'policy': 'sampling.GuidedPolicy',
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
        'prefix': 'plans/plan-0_1',
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
        'horizon': 32,
        'n_diffusion_steps': 32,
    },
}