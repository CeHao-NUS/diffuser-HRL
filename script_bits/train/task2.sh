export WANDB_API_KEY='8c2ff814e2acd0cb8e3076194610c8cc46daa3f8'


python script_bits/train/train_diffuser.py --dataset tamp_easy_2e4 

python script_bits/train/train_diffuser.py --dataset tamp_med_2e4 

python script_bits/train/train_diffuser.py --dataset tamp_hard_2e4 # 

python script_bits/train/train_diffuser.py --dataset fix8_tamp_easy_2e4 

python script_bits/train/train_diffuser.py --dataset fix8_tamp_med_2e4  #

python script_bits/train/train_diffuser.py --dataset fix8_tamp_hard_2e4 #

# eval train

python script_bits/train/train_diffuser.py --dataset tamp_easy_2e4_eval

python script_bits/train/train_diffuser.py --dataset tamp_med_2e4_eval

python script_bits/train/train_diffuser.py --dataset tamp_hard_2e4_eval # 

python script_bits/train/train_diffuser.py --dataset fix8_tamp_easy_2e4_eval

python script_bits/train/train_diffuser.py --dataset fix8_tamp_med_2e4_eval #

python script_bits/train/train_diffuser.py --dataset fix8_tamp_hard_2e4_eval #

# real eval plan

python script_bits/eval/plan.py --dataset tamp_easy_2e4_eval

python script_bits/eval/plan.py --dataset tamp_med_2e4_eval

python script_bits/eval/plan.py --dataset tamp_hard_2e4_eval

python script_bits/eval/plan.py --dataset fix8_tamp_easy_2e4_eval

python script_bits/eval/plan.py --dataset fix8_tamp_med_2e4_eval

python script_bits/eval/plan.py --dataset fix8_tamp_hard_2e4_eval

# ======================================== 5e4 ========================================

python script_bits/train/train_diffuser.py --dataset tamp_easy_5e4 

python script_bits/train/train_diffuser.py --dataset fix8_tamp_easy_5e4 


# eval train

python script_bits/train/train_diffuser.py --dataset tamp_easy_5e4_eval

python script_bits/train/train_diffuser.py --dataset fix8_tamp_easy_5e4_eval

# real eval plan

python script_bits/eval/plan.py --dataset tamp_easy_5e4_eval

python script_bits/eval/plan.py --dataset fix8_tamp_easy_5e4_eval

