

# 0. varh (no guided) =========================================================================
python scripts/train/train_diffuser.py --config 'config.single.varh.train_diff_var1'

python scripts/single/plan.py --config 'config.single.varh.plan_diff_var1' 

# 1. fix + varh_guided =========================================================================
python scripts/train/train_diffuser.py --config 'config.single.train_diff' 
python scripts/train/train_value.py --config 'config.single.varh.train_diff_var1'

python scripts/single/plan.py --config 'config.single.varh.plan_Dfix_Gvar1'


# 2. varh + varh_guided =========================================================================
python scripts/train/train_diffuser.py --config 'config.single.varh.plan_diff_var1'



# store list =================================================================
python scripts/single/plan_list.py --config 'config.single.plan_diff_store'  \
 --conditional True --init_pose "(3,1)" --target "(1, 1)"

python scripts/single/plan_list.py --config 'config.single.plan_guided_store'  \
 --conditional True --init_pose "(1,2)" --target "(1, 1)" --prefix 'plans/guided_store/'
