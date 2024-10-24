from config.single.plan_diff import *

base['plan'].update({
    'value_loadpath': 'f:values/single_varh1_H{horizon}_T{n_diffusion_steps}_d{discount}_mH{min_horizon}',
    'value_epoch': 'latest',
    'prefix': 'plans/single_Dfix_Gvar1/',
    'min_horizon': 16,
})

