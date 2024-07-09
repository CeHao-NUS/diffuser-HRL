
# change init pose / goal
# suffix: 


python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1, 1)" --target "(3,1)" --suffix "far"

python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1, 1)" --target "(3,3)" --suffix "turn1"

python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1, 1)" --target "(1,3)" --suffix "line"

python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1, 1)" --target "(1,2)" --suffix "near"

# 

# =================================================================================================
# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-umaze-v1 \
# --diffusion 'models.SoftGaussianDiffusion'