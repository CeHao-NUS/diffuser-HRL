# p-3_0: HL_fix + LL_fix


from diffuser.utils import watch
import diffuser.sampling as sampling

plan_args_to_watch = [
    ('prefix', ''),
    ##
    ('HL_horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    # ('value_horizon', 'V'),
    # ('discount', 'd'),
    ('normalizer', ''),
    # ('batch_size', 'b'),
    ('seg_length', 'L'),
    ##
    ('conditional', 'cond'),
]



base = {
    'plan': {
        'LL_diffusion_loadpath': 'f:diffusion/LL_diffuser_varh1_H{LL_horizon}_T{n_diffusion_steps}_mH{LL_min_horizon}',
        'LL_diffusion_epoch': 'latest',
        'LL_value_loadpath': 'f:values/LL_varh1_H{LL_horizon}_T{n_diffusion_steps}_d{discount}_mH{LL_min_horizon}',
        'LL_value_epoch': 'latest',

        'HL_diffusion_loadpath': 'f:diffusion/HL_diffuser_H{HL_horizon}_T{n_diffusion_steps}_D{downsample}',
        'HL_diffusion_epoch': 'latest',
        'HL_value_loadpath': None,
        'HL_value_epoch': None,

        'downsample': 16,
        'seg_length': 11,

        'LL_min_horizon': 1, ## for LL

        'HL_batch_size': 1,
        'LL_goal_reach': False,

        'HL_guide': 'sampling.ValueGuide',
        'HL_policy': 'sampling.GuidedPolicy',
        'LL_guide': 'sampling.ValueGuide',
        'LL_policy': 'sampling.GuidedPolicy',

        'max_episode_length': 1000,
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': None,

        ## sample_kwargs
        'n_guide_steps': 2,
        'LL_scale': 0.1,
        'HL_scale': 1.0,
        't_stopgrad': 2,
        'scale_grad_by_std': True,
        'LL_sample_fn': sampling.adj_cond_guided_p_sample,

        ## serialization
        'loadbase': None,
        'logbase': 'logs',
        'prefix': 'plans/hier/',
        'exp_name': watch(plan_args_to_watch),
        'vis_freq': 10,
        'max_render': 8,

        ## diffusion model
        'horizon': 320,
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
        'HL_horizon': 248,
        'n_diffusion_steps': 32,
        'seg_length': 5, # 31*4 = 124
        'downsample': 31,
    },
}

maze2d_medium_v1 = {
    'plan': {
        'LL_horizon': 16,
        'HL_horizon': 300,
        'n_diffusion_steps': 32,
        'seg_length': 18, # 17*15 = 255
        'downsample': 15,
    },
}

maze2d_large_v1 = {
    'plan': {
        'LL_horizon': 32,
        'HL_horizon': 384,
        'n_diffusion_steps': 32,
        'seg_length': 11,
        'downsample': 32,
    },
}