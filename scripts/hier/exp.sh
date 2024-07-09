

python scripts/hier/train_hl.py --config config.hier.maze2d_hl \
   --dataset maze2d-umaze-v1 

python scripts/hier/train_ll.py --config config.hier.maze2d_ll \
   --dataset maze2d-umaze-v1 




#   ================================= hl + ll

python scripts/hier/plan_maze2d.py \
--conditional True --init_pose "(1, 1)" --target "(3,1)" --suffix "far_l3" \
--goal_length 3

python scripts/hier/plan_maze2d.py \
--conditional True --init_pose "(1, 1)" --target "(3,3)" --suffix "turn1_l3" \
--goal_length 3

python scripts/hier/plan_maze2d.py \
--conditional True --init_pose "(1, 1)" --target "(1,3)" --suffix "line" \
--goal_length 4

python scripts/hier/plan_maze2d.py \
--conditional True --init_pose "(1, 1)" --target "(1,2)" --suffix "near" \
--goal_length 3

python scripts/hier/plan_maze2d.py \
--conditional True --init_pose "(1, 1)" --target "(1,1.5)" --suffix "near1.5" \
--goal_length 2

python scripts/hier/plan_maze2d.py \
--conditional True --init_pose "(1, 1)" --target "(1,1.1)" --suffix "near1.1" \
--goal_length 2

# ================================= hl test
python scripts/hier/plan_maze2d_hl.py --config config.hier.maze2d_hl \
  --dataset maze2d-umaze-v1 

python scripts/hier/plan_maze2d_hl.py --config config.hier.maze2d_hl \
  --dataset maze2d-umaze-v1 \
  --conditional True --init_pose "(1, 1)" --target "(3,1)" --suffix "far"

python scripts/hier/plan_maze2d_hl.py --config config.hier.maze2d_hl \
  --dataset maze2d-umaze-v1 \
  --conditional True --init_pose "(1, 1)" --target "(3,3)" --suffix "turn1"

python scripts/hier/plan_maze2d_hl.py --config config.hier.maze2d_hl \
  --dataset maze2d-umaze-v1 \
  --conditional True --init_pose "(1, 1)" --target "(1,3)" --suffix "line"

python scripts/hier/plan_maze2d_hl.py --config config.hier.maze2d_hl \
  --dataset maze2d-umaze-v1 \
  --conditional True --init_pose "(1, 1)" --target "(1,2)" --suffix "near"

