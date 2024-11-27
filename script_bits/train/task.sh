
# bash script_bits/train/task.sh

export WANDB_API_KEY='8c2ff814e2acd0cb8e3076194610c8cc46daa3f8'

python script_bits/train/train_diffuser.py --dataset tamp_zero --device "cuda:7"

python script_bits/train/train_diffuser.py --dataset tamp_easy --device "cuda:7"

python script_bits/train/train_diffuser.py --dataset tamp_med --device "cuda:6"

python script_bits/train/train_diffuser.py --dataset tamp_hard --device "cuda:5"

python script_bits/train/train_diffuser.py --dataset tamp_super --device "cuda:4"



# 
python script_bits/eval/plan.py --dataset tamp_zero --device "cuda:7"

python script_bits/eval/plan.py --dataset tamp_easy --device "cuda:7"

python script_bits/eval/plan.py --dataset tamp_med --device "cuda:7"

python script_bits/eval/plan.py --dataset tamp_hard --device "cuda:5"



# create for eval ====================

python script_bits/train/train_diffuser.py --dataset tamp_zero_eval --device "cuda:7"

python script_bits/train/train_diffuser.py --dataset tamp_easy_eval --device "cuda:7"

python script_bits/train/train_diffuser.py --dataset tamp_med_eval --device "cuda:7"

python script_bits/train/train_diffuser.py --dataset tamp_hard_eval --device "cuda:5"

# eval
python script_bits/eval/plan.py --dataset tamp_zero_eval --device "cuda:7"

python script_bits/eval/plan.py --dataset tamp_easy_eval --device "cuda:7"

python script_bits/eval/plan.py --dataset tamp_med_eval --device "cuda:7"

python script_bits/eval/plan.py --dataset tamp_hard_eval --device "cuda:5"


# ========================== extra test ==========================

python script_bits/train/train_diffuser.py --dataset tamp_78 --device "cuda:6"

python script_bits/train/train_diffuser.py --dataset tamp_77 --device "cuda:5"

python script_bits/train/train_diffuser.py --dataset tamp_fix --device "cuda:4"


python script_bits/eval/plan.py --dataset tamp_78 --device "cuda:6"

python script_bits/eval/plan.py --dataset tamp_77 --device "cuda:5"

python script_bits/eval/plan.py --dataset tamp_fix --device "cuda:4"

# create
python script_bits/train/train_diffuser.py --dataset tamp_78_eval --device "cuda:6"

python script_bits/train/train_diffuser.py --dataset tamp_77_eval --device "cuda:5"

python script_bits/train/train_diffuser.py --dataset tamp_fix_eval --device "cuda:4"

# eval