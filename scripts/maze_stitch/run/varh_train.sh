

python scripts/maze_stitch/train/train_diffuser.py --config 'config.stitch.train.train_LL_varh' --dataset maze2d-umaze-v1
python scripts/maze_stitch/train/train_value.py --config 'config.stitch.train.train_LL_varh' --dataset maze2d-umaze-v1

python scripts/maze_stitch/train/train_diffuser.py --config 'config.stitch.train.train_LL_varh' --dataset maze2d-medium-v1
# python scripts/maze_stitch/train/train_value.py --config 'config.stitch.train.train_LL_varh' --dataset maze2d-medium-v1

python scripts/maze_stitch/train/train_diffuser.py --config 'config.stitch.train.train_LL_varh' --dataset maze2d-large-v1
# python scripts/maze_stitch/train/train_value.py --config 'config.stitch.train.train_LL_varh' --dataset maze2d-large-v1