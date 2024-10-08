# env_list = ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1']


# train diffusion
python scripts/train/train_diffuser.py --config 'config.single.train_diff'  --dataset maze2d-umaze-v1 --device "cuda"


# plan traj
python scripts/single/plan.py --config 'config.single.plan_diff'  --dataset maze2d-umaze-v1 --device "cuda"



python scripts/train/train_diffuser.py --config 'config.single.train_diff'  --dataset maze2d-medium-v1 --device "cuda:1"
python scripts/train/train_diffuser.py --config 'config.single.train_diff'  --dataset maze2d-large-v1 --device "cuda:2"




# debug

python scripts/train/train_diffuser.py --config 'config.single.train_diff_test'  --dataset maze2d-test-v0 --device "cuda:7"

