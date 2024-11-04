from config.hier.plan_diff import *

base['plan'].update({

    'HL_diffusion_loadpath': 'f:diffusion/HL_diff_varh_H{HL_horizon}_T{n_diffusion_steps}_D{downsample}',
    'prefix': 'plans/hier_varh/',
})