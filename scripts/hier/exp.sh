

python scripts/hier/train_hl.py --config config.hier.maze2d_hl \
   --dataset maze2d-umaze-v1 

python scripts/hier/train_ll.py --config config.hier.maze2d_ll \
   --dataset maze2d-umaze-v1 


# =================================
python scripts/hier/plan_maze2d.py 

# =================================

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

