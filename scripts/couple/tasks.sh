export WANDB_API_KEY='8c2ff814e2acd0cb8e3076194610c8cc46daa3f8'

python scripts/train/train_diffuser.py --config 'config.couple.train_diff_HL_fixgap' --dataset maze2d-umaze-v1 --device "cuda:7"
python scripts/train/train_diffuser.py --config 'config.couple.train_diff_HL_fixgap' --dataset maze2d-medium-v1 --device "cuda:7"
python scripts/train/train_diffuser.py --config 'config.couple.train_diff_HL_fixgap' --dataset maze2d-large-v1 --device "cuda:5"


python scripts/couple/plan.py --config 'config.couple.plan_diff_HLGap' --dataset maze2d-umaze-v1 --device "cuda"
python scripts/couple/plan.py --config 'config.couple.plan_diff_HLGap' --dataset maze2d-medium-v1 --device "cuda"
python scripts/couple/plan.py --config 'config.couple.plan_diff_HLGap' --dataset maze2d-large-v1 --device "cuda"

# value

python scripts/train/train_value.py --config 'config.couple.train_diff_HL_fixgap' --dataset maze2d-large-v1 --device "cuda:6"



# plan the LL whole traj

python scripts/couple/plan_LL_whole.py --config 'config.couple.plan_diff_LL_whole'  --dataset maze2d-umaze-v1 --device "cuda:7"

