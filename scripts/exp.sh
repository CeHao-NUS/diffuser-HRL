
# change init pose / goal
# suffix: 


# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-umaze-v1 \
# --conditional True --init_pose "(1, 1)" --target "(3,1)" --suffix "far1"

# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-umaze-v1 \
# --conditional True --init_pose "(1, 1)" --target "(3,2)" --suffix "far2"

# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-umaze-v1 \
# --conditional True --init_pose "(1, 1)" --target "(3,1.5)" --suffix "far1.5"

# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-umaze-v1 \
# --conditional True --init_pose "(1, 1)" --target "(3,3)" --suffix "turn1"

# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-umaze-v1 \
# --conditional True --init_pose "(1, 1)" --target "(1,3)" --suffix "line"

python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1, 1)" --target "(1,2)" --suffix "near2"

python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1, 1)" --target "(1,1.5)" --suffix "near1.5"

python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1, 1)" --target "(1,1.1)" --suffix "near1.1"

# =================================================================================================

# med
# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-medium-v1 \
# --conditional True --init_pose "(1, 1)" --target "(6,6)" --suffix "far1"

# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-medium-v1 \
# --conditional True --init_pose "(1, 1)" --target "(2,2)" --suffix "near"

# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 \
# --conditional True --init_pose "(1, 1)" --target "(7,10)" --suffix "far1"


# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 \
# --conditional True --init_pose "(1, 1)" --target "(1,3)" --suffix "near"

# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 \
# --conditional True --init_pose "(1, 1)" --target "(3,1)" --suffix "near31"

# python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 \
# --conditional True --init_pose "(1, 1)" --target "(5,4)" --suffix "near54"