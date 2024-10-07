


# train LL diff
python scripts/maze_stitch/train/train_diffuser.py --config 'config.hier.train_LL' --dataset maze2d-large-v1 --device "cuda"

# train HL_diff
python scripts/maze_stitch/train/train_diffuser.py --config 'config.hier.train_HL' --dataset maze2d-large-v1 --device "cuda"

# plan diff
python scripts/maze_stitch/plan/plan_3.py --config 'config.hier.plan_diff'
