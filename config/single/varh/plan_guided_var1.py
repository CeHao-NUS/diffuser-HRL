from config.single.varh.plan_diff_var1 import *

base['plan'].update({
    'value_loadpath': 'f:values/single_varh1_H{horizon}_T{n_diffusion_steps}_d{discount}_mH{min_horizon}',
    'value_epoch': 'latest',
    'prefix': 'plans/single_guided_varh1/',
})

