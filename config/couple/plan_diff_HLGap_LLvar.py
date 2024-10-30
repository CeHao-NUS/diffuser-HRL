from config.couple.plan_diff_HLGap import *

base['plan'].update({
    'LL_diffusion_loadpath': 'f:diffusion/LL_diffuser_varh1_H{LL_horizon}_T{n_diffusion_steps}_mH{LL_min_horizon}',
    'LL_min_horizon': 1,

    'prefix': 'plans/couple/HLGap_LLvarh',
})