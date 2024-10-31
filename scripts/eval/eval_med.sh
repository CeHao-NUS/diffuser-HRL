
# bash scripts/eval/eval_med.sh

# single
# task="scripts/single/plan.py"
# config="config.single.plan_diff"
# config="config.single.varh.plan_diff_var1"

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

# --target "(6,5)" # hard (6,6) easy

for init_pose in "(6,1)" "(5,1)" "(2,4)" "(3,2)" "(4,1)" "(4,5)" "(5,3)" "(1,5)" "(4,6)"
do
    python $task --config $config --dataset maze2d-medium-v1 --device "cuda" \
    --conditional True --init_pose $init_pose --target "(6,5)" --suffix $init_pose   \
      &
done

wait

