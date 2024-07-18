python scripts/varh/plan_maze2d.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1, 1)" --target "(3,1)" --suffix "far1" --min_horizon 96

python scripts/varh/plan_maze2d.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1, 1)" --target "(3,2)" --suffix "far2"  --min_horizon 96

python scripts/varh/plan_maze2d.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1, 1)" --target "(3,1.5)" --suffix "far1.5"  --min_horizon 96

python scripts/varh/plan_maze2d.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1, 1)" --target "(3,3)" --suffix "turn1"  --min_horizon 96

python scripts/varh/plan_maze2d.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1, 1)" --target "(1,3)" --suffix "line"  --min_horizon 96

python scripts/varh/plan_maze2d.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1, 1)" --target "(1,2)" --suffix "near2"  --min_horizon 96

python scripts/varh/plan_maze2d.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1, 1)" --target "(1,1.5)" --suffix "near1.5"  --min_horizon 96

python scripts/varh/plan_maze2d.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1, 1)" --target "(1,1.1)" --suffix "near1.1"  --min_horizon 96
