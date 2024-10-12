# env_list = ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1']


# train diffusion
python scripts/train/train_diffuser.py --config 'config.single.train_diff'  --dataset maze2d-umaze-v1 --device "cuda"
# plan traj
python scripts/single/plan.py --config 'config.single.plan_diff'  --dataset maze2d-umaze-v1 --device "cuda"

# med, large
python scripts/train/train_diffuser.py --config 'config.single.train_diff'  --dataset maze2d-medium-v1 --device "cuda:1"
python scripts/train/train_diffuser.py --config 'config.single.train_diff'  --dataset maze2d-large-v1 --device "cuda:2"


# test big
python scripts/train/train_diffuser.py --config 'config.single.train_diff_test'  --dataset maze2d-testbig-v0 --device "cuda:7"
python scripts/single/plan.py --config 'config.single.plan_diff_test'  --dataset maze2d-testbig-v0 --device "cuda:0"


#  CoupledGaussianDiffusion_ForwardNoise
python scripts/train/train_diffuser.py --config 'config.single.train_diff_coupled_forwardnoise'  --dataset maze2d-umaze-v1 --device "cuda:4" \
 --prefix 'diffusion/test_noise_postcond_batch128' --batch_size 128

python scripts/train/train_diffuser.py --config 'config.single.train_diff_coupled_forwardnoise'  --dataset maze2d-medium-v1 --device "cuda:6"
python scripts/train/train_diffuser.py --config 'config.single.train_diff_coupled_forwardnoise'  --dataset maze2d-large-v1 --device "cuda:7"

python scripts/single/plan.py --config 'config.single.plan_diff_coupled_forwardnoise'  --dataset maze2d-umaze-v1 --device "cuda:4"

# train with varh datasets

python scripts/train/train_diffuser.py --config 'config.single.varh.train_diff_var1'  --dataset maze2d-umaze-v1 --device "cuda:7"
python scripts/train/train_diffuser.py --config 'config.single.varh.train_diff_var1'  --dataset maze2d-medium-v1 --device "cuda:6"
python scripts/train/train_diffuser.py --config 'config.single.varh.train_diff_var1'  --dataset maze2d-large-v1 --device "cuda:5"

python scripts/train/train_diffuser.py --config 'config.single.varh.train_diff_var2'  --dataset maze2d-umaze-v1 --device "cuda:7"
python scripts/train/train_diffuser.py --config 'config.single.varh.train_diff_var2'  --dataset maze2d-medium-v1 --device "cuda:6"
python scripts/train/train_diffuser.py --config 'config.single.varh.train_diff_var2'  --dataset maze2d-large-v1 --device "cuda:5"



# train a dummy condition in training
python scripts/train/train_diffuser.py --config 'config.single.train_diff'  --dataset maze2d-umaze-v1 --device "cuda:1" \
 --loader "datasets.DummyGoalDataset" --prefix 'diffusion/dummygoal_single_diffuser'

python scripts/single/plan.py --config 'config.single.plan_diff'  --dataset maze2d-umaze-v1 --device "cuda" \
 --diffusion_loadpath 'f:diffusion/dummygoal_single_diffuser_H{horizon}_T{n_diffusion_steps}' --prefix 'plans/dummygoal_single/'

python scripts/single/plan.py --config 'config.single.plan_diff_coupled_forwardnoise'  --dataset maze2d-umaze-v1 --device "cuda" \
 --diffusion_loadpath 'f:diffusion/dummygoal_single_diffuser_H{horizon}_T{n_diffusion_steps}' --prefix 'plans/dummygoal_single_fornoise/'

# test learn epsilon, not bad
python scripts/train/train_diffuser.py --config 'config.single.train_diff'  --dataset maze2d-umaze-v1 --device "cuda:4" \
 --predict_epsilon True --prefix 'diffusion/ep_single/'


# do the single store plotting
python scripts/train/train_diffuser.py --config 'config.single.train_diff_store'  --dataset maze2d-umaze-v1 --device "cuda:4"
python scripts/single/plan_list.py --config 'config.single.plan_diff_store'  --dataset maze2d-umaze-v1 --device "cuda:4"

