# env_list = ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1']


# train LL diff
python scripts/train/train_diffuser.py --config 'config.hier.train_diff_LL' --dataset maze2d-umaze-v1 --device "cuda"

# train HL_diff
python scripts/train/train_diffuser.py --config 'config.hier.train_diff_HL' --dataset maze2d-umaze-v1 --device "cuda"

# plan diff
python scripts/hier/plan.py --config 'config.hier.plan_diff' --dataset maze2d-umaze-v1 --device "cuda"


# med
python scripts/train/train_diffuser.py --config 'config.hier.train_diff_LL' --dataset maze2d-medium-v1 --device "cuda:3"
python scripts/train/train_diffuser.py --config 'config.hier.train_diff_HL' --dataset maze2d-medium-v1 --device "cuda:4"

python scripts/train/train_diffuser.py --config 'config.hier.train_diff_LL' --dataset maze2d-large-v1 --device "cuda:5"
python scripts/train/train_diffuser.py --config 'config.hier.train_diff_HL' --dataset maze2d-large-v1 --device "cuda:6"

