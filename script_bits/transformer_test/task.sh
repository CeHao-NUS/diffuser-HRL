export WANDB_API_KEY='8c2ff814e2acd0cb8e3076194610c8cc46daa3f8'


# =========== train ===========
python script_bits/transformer_test/train.py --dataset tamp_easy_78 --device "cuda:6"

python script_bits/transformer_test/train.py --dataset tamp_med_78 

python script_bits/transformer_test/train.py --dataset tamp_hard_78


# =========== eval ===========
python script_bits/transformer_test/eval_para.py --dataset tamp_easy_78 --device "cuda:3"

python script_bits/transformer_test/eval_para.py --dataset tamp_med_78 --device "cuda:2"

python script_bits/transformer_test/eval_para.py --dataset tamp_hard_78 --device "cuda:1"

# ===== eval dataset =====
# create

python script_bits/transformer_test/train.py --dataset tamp_easy_78_eval --device "cuda:7"

python script_bits/transformer_test/train.py --dataset tamp_med_78_eval

python script_bits/transformer_test/train.py --dataset tamp_hard_78_eval

# eval
python script_bits/transformer_test/eval_para.py --dataset tamp_easy_78_eval --device "cuda:6"

python script_bits/transformer_test/eval_para.py --dataset tamp_med_78_eval

python script_bits/transformer_test/eval_para.py --dataset tamp_hard_78_eval --device "cuda:4"

# ===================== dyanmic =====================
python script_bits/transformer_test/dynamic_eval.py --dataset tamp_easy_78

python script_bits/transformer_test/dynamic_eval.py --dataset tamp_easy_78_eval

python script_bits/transformer_test/dynamic_eval.py --dataset tamp_med_78

python script_bits/transformer_test/dynamic_eval.py --dataset tamp_med_78_eval

python script_bits/transformer_test/dynamic_eval.py --dataset tamp_hard_78

python script_bits/transformer_test/dynamic_eval.py --dataset tamp_hard_78_eval


# ================== for fix_8 ==================
# =========== train ===========
python script_bits/transformer_test/train.py --dataset fix8_tamp_easy_78 --device "cuda:3"

python script_bits/transformer_test/train.py --dataset fix8_tamp_med_78 --device "cuda:2"

python script_bits/transformer_test/train.py --dataset fix8_tamp_hard_78 --device "cuda:1"

# =========== eval ===========
python script_bits/transformer_test/eval_para.py --dataset fix8_tamp_easy_78 --device "cuda:7"

python script_bits/transformer_test/eval_para.py --dataset fix8_tamp_med_78 --device "cuda:6"

python script_bits/transformer_test/eval_para.py --dataset fix8_tamp_hard_78 --device "cuda:5"

# ===== eval dataset =====
# create

python script_bits/transformer_test/train.py --dataset fix8_tamp_easy_78_eval --device "cuda:7"

python script_bits/transformer_test/train.py --dataset fix8_tamp_med_78_eval

python script_bits/transformer_test/train.py --dataset fix8_tamp_hard_78_eval

# eval

python script_bits/transformer_test/eval_para.py --dataset fix8_tamp_easy_78_eval --device "cuda:6"

python script_bits/transformer_test/eval_para.py --dataset fix8_tamp_med_78_eval

python script_bits/transformer_test/eval_para.py --dataset fix8_tamp_hard_78_eval --device "cuda:4"

# ===================== dyanmic =====================

python script_bits/transformer_test/dynamic_eval.py --dataset fix8_tamp_easy_78_eval

python script_bits/transformer_test/dynamic_eval.py --dataset fix8_tamp_med_78_eval

python script_bits/transformer_test/dynamic_eval.py --dataset fix8_tamp_hard_78_eval

