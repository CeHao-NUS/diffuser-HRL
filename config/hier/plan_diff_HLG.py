# HL guided

from config.hier.plan_diff import *


# TODO

base['plan'].update({
    'HL_diffusion_loadpath': 'f:diffusion/HL_varh_diffuser_H{HL_horizon}_T{n_diffusion_steps}_D{downsample}_mH{HL_min_horizon}',
    'HL_diffusion_epoch': 'latest',
    'HL_value_loadpath': 'f:values/HL_varh_value_H{HL_horizon}_T{n_diffusion_steps}_d{discount}_D{downsample}_mH{HL_min_horizon}',
    'HL_value_epoch': 'latest',
    'HL_min_horizon': 1,
    'HL_batch_size': 10,

    'prefix': 'plans/plan_3_1/'
})