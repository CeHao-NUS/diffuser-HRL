

# 0. varh (no guided) =========================================================================
python scripts/train/train_diffuser.py --config 'config.single.varh.train_diff_var1'

python scripts/single/plan.py --config 'config.single.varh.plan_diff_var1' 

# 1. fix + varh_guided =========================================================================
python scripts/train/train_diffuser.py --config 'config.single.train_diff' 
python scripts/train/train_value.py --config 'config.single.varh.train_diff_var1'

python scripts/single/plan.py --config 'config.single.varh.plan_Dfix_Gvar1'


# 2. varh + varh_guided =========================================================================
python scripts/train/train_diffuser.py --config 'config.single.varh.plan_diff_var1'



