
# change init pose / goal
# suffix: 

python scripts/plan_maze2d.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(3,1)" --target "(1, 1)" --suffix "far1" 

python scripts/plan_maze2d.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(3,2)" --target "(1, 1)" --suffix "far2" 

python scripts/plan_maze2d.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(3,1.5)" --target "(1, 1)" --suffix "far1.5" 

python scripts/plan_maze2d.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(3,3)" --target "(1, 1)" --suffix "turn1" 

python scripts/plan_maze2d.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1,3)" --target "(1, 1)" --suffix "line" 

python scripts/plan_maze2d.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1,2)" --target "(1, 1)" --suffix "near2" 

python scripts/plan_maze2d.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1,1.5)" --target "(1, 1)" --suffix "near1.5" 

python scripts/plan_maze2d.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1,1.1)" --target "(1, 1)" --suffix "near1.1" 



# python scripts/varh/plan_maze2d.py --dataset maze2d-umaze-v1 \
# --conditional True --init_pose "(2,3)" --target "(1,2)" --suffix "test"

# python scripts/varh/plan_maze2d.py --dataset maze2d-umaze-v1 \
# --conditional True --init_pose "(3,1)" --target "(3, 1.2)" --suffix "test2"

# python scripts/plan_maze2d.py --dataset maze2d-umaze-v1 \
# --conditional True --init_pose "(3,1)" --target "(3, 1.2)" --suffix "test2"

# =================================================================================================

# # med
# for plan_horizon in 64 128 192 256; 
# do
    # python scripts/plan_maze2d.py --dataset maze2d-medium-v1 \
    # --conditional True --init_pose "(1, 1)" --target "(6,6)" --suffix "far66" \
    # --plan_horizon $plan_horizon

    # python scripts/plan_maze2d.py --dataset maze2d-medium-v1 \
    # --conditional True --init_pose "(1, 1)" --target "(2,2)" --suffix "near22" \
    # --plan_horizon $plan_horizon

    # python scripts/plan_maze2d.py --dataset maze2d-medium-v1 \
    # --conditional True --init_pose "(1, 1)" --target "(3,3)" --suffix "near33" \
    # --plan_horizon $plan_horizon

    # python scripts/plan_maze2d.py --dataset maze2d-medium-v1 \
    # --conditional True --init_pose "(1, 1)" --target "(4,2)" --suffix "near42" \
    # --plan_horizon $plan_horizon

    # python scripts/plan_maze2d.py --dataset maze2d-medium-v1 \
    # --conditional True --init_pose "(1, 1)" --target "(3,4)" --suffix "near34" \
    # --plan_horizon $plan_horizon

    # python scripts/plan_maze2d.py --dataset maze2d-medium-v1 \
    # --conditional True --init_pose "(1, 1)" --target "(1,6)" --suffix "to16" \
    # --plan_horizon $plan_horizon

    # python scripts/plan_maze2d.py --dataset maze2d-medium-v1 \
    # --conditional True --init_pose "(1, 1)" --target "(6,3)" --suffix "to63" \
    # --plan_horizon $plan_horizon

# done
















# python scripts/plan_maze2d.py --dataset maze2d-large-v1 \
# --conditional True --init_pose "(1, 1)" --target "(7,10)" --suffix "far1"


# python scripts/plan_maze2d.py --dataset maze2d-large-v1 \
# --conditional True --init_pose "(1, 1)" --target "(1,3)" --suffix "near"

# python scripts/plan_maze2d.py --dataset maze2d-large-v1 \
# --conditional True --init_pose "(1, 1)" --target "(3,1)" --suffix "near31"

# python scripts/plan_maze2d.py --dataset maze2d-large-v1 \
# --conditional True --init_pose "(1, 1)" --target "(5,4)" --suffix "near54"