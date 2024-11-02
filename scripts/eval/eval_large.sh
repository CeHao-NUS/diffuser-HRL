
# bash scripts/eval/eval_large.sh

# single
# task="scripts/single/plan.py"
# config="config.single.plan_diff"
# config="config.single.varh.plan_diff_var1"

# config="config.single.plan_guided"
# config="config.single.varh.plan_guided_var1"

# single store
# task="scripts/single/plan_list.py"
# config="config.single.plan_diff_store"
# config="config.single.plan_guided_store"

# hier
# task="scripts/hier/plan.py"
# config='config.hier.plan_diff'

# couple
task="scripts/couple/plan.py"
# config='config.couple.plan_diff_HLGap'
config='config.couple.plan_diff_HLGap_LLvar'

for init_pose in "(1,1)" "(1,4)" "(1,9)" "(7,1)" "(3,8)" "(5,4)" "(5,7)" "(3,6)" "(6,8)" "(5,10)"
do
    python $task --config $config --dataset maze2d-large-v1 --device "cuda" \
    --conditional True --init_pose $init_pose --target "(7,9)" --suffix $init_pose  \
     &
done

wait

