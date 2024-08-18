# # p-1_2: LL_varh + guide

from config.stitch.plan.plan_1_1 import *
base['plan']['value_loadpath'] = 'f:values/LL_varh_value_H{horizon}_T{n_diffusion_steps}_d{discount}_mH{min_horizon}'
base['plan']['value_epoch'] = 'latest'
