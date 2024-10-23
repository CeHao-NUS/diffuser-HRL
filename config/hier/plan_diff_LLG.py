# LL guided

from config.hier.plan_diff import *

base['plan'].update({
    'LL_value_loadpath': 'f:values/LL_diffuser_H{LL_horizon}_T{n_diffusion_steps}_d{discount}',
    'LL_value_epoch': 'latest',
    'prefix': 'plans/hier_LLG/'
})
