
python scripts/train/train_value.py --config 'config.single.train_diff' --dataset maze2d-umaze-v1

python scripts/train/train_value.py --config 'config.single.train_diff' --dataset maze2d-medium-v1

# plan guided =================================================================
python scripts/single/plan.py --config 'config.single.plan_guided' --dataset maze2d-umaze-v1 --device "cuda" \
 --conditional True --init_pose "(3, 1)" --target "(1, 1)"


python scripts/single/plan.py --config 'config.single.plan_guided' --dataset maze2d-medium-v1 --device "cuda" \
    --conditional True --init_pose "(6, 1)" --target "(6, 5)"



# store list =================================================================

python scripts/single/plan_list.py --config 'config.single.plan_guided_store' --dataset maze2d-medium-v1 --device "cuda" \
    --conditional True --init_pose "(6, 1)" --target "(6, 5)"

