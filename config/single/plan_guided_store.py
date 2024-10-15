from config.single.plan_diff_store import *

base['plan'].update({
    'value_loadpath': 'f:values/single_diffuser_H{horizon}_T{n_diffusion_steps}_d{discount}',
    'value_epoch': 'latest',
    'prefix': 'plans/single_guided_store/',
})