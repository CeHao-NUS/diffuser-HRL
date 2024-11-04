from config.couple.plan_diff_HLGap_LLvar import *

base['plan'].update({
    'HL_value_loadpath': 'f:values/HL_gap_value_H{HL_horizon}_T{n_diffusion_steps}_d{discount}_mH{min_horizon}_L{seg_length}',
    'HL_value_epoch': 'latest',

    'prefix': 'plans/couple/HLGapGuided_LLvarh',

    'HL_scale': -1,
})