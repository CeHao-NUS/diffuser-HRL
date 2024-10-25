from config.single.plan_diff_store import *

# base['plan'].update({
#     'value_loadpath': 'f:values/single_diffuser_H{horizon}_T{n_diffusion_steps}_d{discount}',
#     'value_epoch': 'latest',
#     'prefix': 'plans/single_guided_store/',
# })

# from config.single.varh.plan_diff_var1 import *

base['plan'].update({
    'value_loadpath': 'f:values/single_varh1_H{horizon}_T{n_diffusion_steps}_d{discount}_mH{min_horizon}',
    'value_epoch': 'latest',
    'prefix': 'plans/single_guided_varh1/',
    'min_horizon': 16,
    'scale': 0.0,
})

