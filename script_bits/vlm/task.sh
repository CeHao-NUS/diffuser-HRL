export OPENAI_API_KEY=''

python script_bits/vlm/main.py --dataset fix8_tamp_easy_78

python script_bits/vlm/main.py --dataset fix8_tamp_med_78

python script_bits/vlm/main.py --dataset fix8_tamp_hard_78

python script_bits/vlm/main.py --dataset tamp_easy_78

python script_bits/vlm/main.py --dataset tamp_med_78

python script_bits/vlm/main.py --dataset tamp_hard_78


python script_bits/vlm/dynamic_eval.py --dataset fix8_tamp_easy_78

python script_bits/vlm/dynamic_eval.py --dataset fix8_tamp_med_78

python script_bits/vlm/dynamic_eval.py --dataset fix8_tamp_hard_78

python script_bits/vlm/dynamic_eval.py --dataset tamp_easy_78

python script_bits/vlm/dynamic_eval.py --dataset tamp_med_78

python script_bits/vlm/dynamic_eval.py --dataset tamp_hard_78


