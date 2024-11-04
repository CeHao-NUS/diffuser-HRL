

# bash scripts/eval/eval_large_more.sh

MAX_JOBS=50
current_jobs=0
# GPU_IDS=("cuda:0" "cuda:1" "cuda:2" "cuda:3" "cuda:4" "cuda:5" "cuda:6" "cuda:7")
GPU_IDS=("cuda:0" "cuda:1" "cuda:2")
NUM_GPUS=${#GPU_IDS[@]} # Number of GPUs (8 in this case)

# single
task="scripts/single/plan.py"
# config="config.single.plan_diff"
config="config.single.varh.plan_diff_var1"

# config="config.single.plan_guided"
# config="config.single.varh.plan_guided_var1"

# single store
# task="scripts/single/plan_list.py"
# config="config.single.plan_diff_store"
# config="config.single.plan_guided_store"

# hier
# task="scripts/hier/plan.py"
# config='config.hier.plan_diff'
# config='config.hier.varh.plan_diff_HLvar'

# couple
# task="scripts/couple/plan.py"
# config='config.couple.plan_diff_HLGap'
# config='config.couple.plan_diff_HLGap_LLvar'
# config='config.couple.plan_diff_HLGapGuided_LLvar'



init_poses=("(1,1)" "(1,2)" "(1,3)" "(1,4)" "(1,6)" "(1,7)" "(1,8)" "(1,9)" "(1,10)" "(2,1)" "(2,4)" "(2,6)" "(2,8)" "(2,10)" "(3,1)" "(3,2)" "(3,3)" "(3,4)" "(3,5)" "(3,6)" "(3,8)" "(3,9)" "(3,10)" "(4,1)" "(4,6)" "(4,10)" "(5,1)" "(5,2)" "(5,4)" "(5,6)" "(5,7)" "(5,8)" "(5,9)" "(5,10)" "(6,2)" "(6,4)" "(6,6)" "(6,8)" "(7,1)" "(7,2)" "(7,4)" "(7,5)" "(7,6)" "(7,8)" "(7,10)")



for init_pose in "${init_poses[@]}"
do
    # Determine which GPU to assign based on the job index
    gpu_idx=$((current_jobs % NUM_GPUS))
    gpu="${GPU_IDS[$gpu_idx]}"

    python $task --config $config --dataset maze2d-large-v1 --device "cuda" \
    --conditional True --init_pose $init_pose --target "(7,9)" --suffix $init_pose  \
     &

    # Increment the current_jobs counter
    current_jobs=$((current_jobs + 1))

    # If the maximum number of parallel jobs is reached, wait for all of them to finish
    if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
        wait
        current_jobs=0
    fi
done

# Wait for any remaining jobs to complete
wait

echo "Done"

# ===========================================


