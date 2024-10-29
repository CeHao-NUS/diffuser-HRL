
python scripts/train/train_diffuser.py --config 'config.couple.train_diff_HL_fixgap' --dataset maze2d-umaze-v1 --device "cuda:5"

python scripts/couple/plan.py --config 'config.couple.plan_diff_HLGap' --dataset maze2d-umaze-v1 --device "cuda"

