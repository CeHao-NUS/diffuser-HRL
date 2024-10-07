


# train diffusion
python scripts/train/train_diffuser.py --config 'config.single.train_diff'  --dataset maze2d-umaze-v1 --device "cuda"


# plan traj
python scripts/single/plan.py --config 'config.single.plan_diff'  --dataset maze2d-umaze-v1 --device "cuda"

