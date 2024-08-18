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
    ('seg_length', 'L'),
    ##
    ('conditional', 'cond'),
]


base = {
    'plan': {

        ## loading
        'diffusion_loadpath': 'f:diffusion/LL_diffuser_H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',

        'guide_LL': None,
        'value_LL': None,
        # 'sample_fun': sampling.stitch_functions.default_sample_fn,
        'sample_fun': sampling.stitch_functions.LL_LL_joint_sample_fn,

        'seg_length': 3,

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
        'prefix': 'plans/plan-1_0/',
        'exp_name': watch(plan_args_to_watch),
        'vis_freq': 10,
        'max_render': 8,

        ## diffusion model
        'horizon': 256,
        'n_diffusion_steps': 256,

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

# original diffuser
maze2d_umaze_v1 = {
    'plan':{
        'horizon': 128,
        'n_diffusion_steps': 64,
    },
}

maze2d_medium_v1 = {
    'plan':{
        'horizon': 256,
        'n_diffusion_steps': 256,
    },
}

maze2d_large_v1 = {
    'plan':{
        'horizon': 384,
        'n_diffusion_steps': 256,
    },
}
