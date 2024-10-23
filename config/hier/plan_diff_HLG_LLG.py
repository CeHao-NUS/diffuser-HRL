# HL and LL both guided

from config.hier.plan_diff import *

# TODO

base['plan'].update({
    'HL_diffusion_loadpath': 'f:diffusion/HL_varh_diffuser_H{HL_horizon}_T{n_diffusion_steps}_D{downsample}_mH{HL_min_horizon}',
    'HL_diffusion_epoch': 'latest',
    'HL_value_loadpath': 'f:values/HL_varh_value_H{HL_horizon}_T{n_diffusion_steps}_d{discount}_D{downsample}_mH{HL_min_horizon}',
    'HL_value_epoch': 'latest',
    'HL_min_horizon': 1,
    'HL_batch_size': 10,

    'LL_diffusion_loadpath': 'f:diffusion/LL_varh_diffuser_H{LL_horizon}_T{n_diffusion_steps}_mH{LL_min_horizon}',
    'LL_diffusion_epoch': 'latest',
    'LL_value_loadpath': 'f:values/LL_varh_value_H{LL_horizon}_T{n_diffusion_steps}_d{discount}_mH{LL_min_horizon}',
    'LL_value_epoch': 'latest',
    'LL_min_horizon': 1,

    'prefix': 'plans/plan_3_3/'
})