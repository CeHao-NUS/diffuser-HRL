


# train LL diff
python scripts/train/train_diffuser.py --config 'config.hier.train_LL' --dataset maze2d-umaze-v1 --device "cuda"

# train HL_diff
python scripts/train/train_diffuser.py --config 'config.hier.train_HL' --dataset maze2d-umaze-v1 --device "cuda"

# plan diff
python scripts/hier/plan.py --config 'config.hier.plan_diff' --dataset maze2d-umaze-v1 --device "cuda"
