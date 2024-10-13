# env_list = ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1']


# train LL diff
python scripts/train/train_diffuser.py --config 'config.hier.train_diff_LL' --dataset maze2d-umaze-v1 --device "cuda:7"

# train HL_diff
python scripts/train/train_diffuser.py --config 'config.hier.train_diff_HL' --dataset maze2d-umaze-v1 --device "cuda:4"

# plan diff
python scripts/hier/plan.py --config 'config.hier.plan_diff' --dataset maze2d-umaze-v1 --device "cuda"


# med
python scripts/train/train_diffuser.py --config 'config.hier.train_diff_LL' --dataset maze2d-medium-v1 --device "cuda:6"
python scripts/train/train_diffuser.py --config 'config.hier.train_diff_HL' --dataset maze2d-medium-v1 --device "cuda:5"

python scripts/train/train_diffuser.py --config 'config.hier.train_diff_LL' --dataset maze2d-large-v1 --device "cuda:4"
python scripts/train/train_diffuser.py --config 'config.hier.train_diff_HL' --dataset maze2d-large-v1 --device "cuda:3"

python scripts/hier/plan.py --config 'config.hier.plan_diff' --dataset maze2d-medium-v1 --device "cuda"

# for big
python scripts/train/train_diffuser.py --config 'config.hier.train_diff_LL_test'  --dataset maze2d-testbig-v0 --device "cuda:6"
python scripts/train/train_diffuser.py --config 'config.hier.train_diff_HL_test'  --dataset maze2d-testbig-v0 --device "cuda:5"
python scripts/hier/plan.py --config 'config.hier.plan_diff_test' --dataset maze2d-testbig-v0 --device "cuda"