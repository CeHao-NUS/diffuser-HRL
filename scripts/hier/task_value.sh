

python scripts/train/train_value.py --config 'config.hier.train_diff_LL' --dataset maze2d-medium-v1

python scripts/hier/plan.py --config 'config.hier.plan_diff_LLG' --dataset maze2d-medium-v1 --device "cuda" \
    --conditional True --init_pose "(1, 6)" --target "(6, 5)"

# store list =================================================================

python scripts/hier/plan_list.py --config 'config.hier.plan_diff_LLG' --dataset maze2d-medium-v1 --device "cuda" \
    --conditional True --init_pose "(1, 6)" --target "(6, 5)"

