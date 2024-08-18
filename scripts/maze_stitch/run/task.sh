
# --dataset maze2d-umaze-v1
# --dataset maze2d-medium-v1
# --dataset maze2d-large-v1


# p-1_0: LL diffuser
python scripts/maze_stitch/train/train_LL.py --config 'config.stitch.train.train_LL_fixh'
python scripts/maze_stitch/plan/plan_1.py --config 'config.stitch.plan.plan_1_0'

# p-1_1: LL_varh
python scripts/maze_stitch/train/train_LL.py --config 'config.stitch.train.train_LL_varh'
python scripts/maze_stitch/plan/plan_1.py --config 'config.stitch.plan.plan_1_1'

# p-1_2: LL_value_varh
python scripts/maze_stitch/train/train_LL.py --config 'config.stitch.train.train_LL_varh'
python scripts/maze_stitch/train/train_LL_value_varh.py --config 'config.stitch.train.train_LL_varh'
python scripts/maze_stitch/plan/plan_1.py --config 'config.stitch.plan.plan_1_2'


