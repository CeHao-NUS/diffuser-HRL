
# bash scripts/eval/eval_umaze.sh

# single
# task="scripts/single/plan.py"
# config="config.single.plan_diff"


# hier
# task="scripts/hier/plan.py"
# config='config.hier.plan_diff'

# forward_noise
task="scripts/single/plan.py"
config='config.single.plan_diff_coupled_forwardnoise'

for init_pose in "(1,1.5)" "(1,1.1)" "(3,1)" "(3,2)" "(3,1.5)" "(3,3)" "(2,3)" "(1,3)" "(1,2)"
do
    python $task --config $config --dataset maze2d-umaze-v1 \
    --conditional True --init_pose $init_pose --target "(1, 1)" --suffix $init_pose &
done

wait