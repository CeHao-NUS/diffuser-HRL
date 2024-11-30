export WANDB_API_KEY='8c2ff814e2acd0cb8e3076194610c8cc46daa3f8'

python script_bits/transformer_test/train.py

python script_bits/transformer_test/eval.py

python script_bits/transformer_test/eval_para.py



python script_bits/transformer_test/eval_more_gpu.py


torchrun --nproc_per_node=8 script_bits/transformer_test/eval_more_gpu.py --savepath /path/to/save --train_data_dir /path/to/data

accelerate launch --num_processes=8  script_bits/transformer_test/eval_more_gpu.py
