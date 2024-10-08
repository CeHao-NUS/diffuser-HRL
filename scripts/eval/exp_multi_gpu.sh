

# bash scripts/eval/exp_multi_gpu.sh

MAX_JOBS=50
current_jobs=0
GPU_IDS=("cuda:0" "cuda:1" "cuda:2" "cuda:3" "cuda:4" "cuda:5" "cuda:6" "cuda:7")
NUM_GPUS=${#GPU_IDS[@]} # Number of GPUs (8 in this case)



# dataset="maze2d-umaze-v1"
# dataset="maze2d-medium-v1"
dataset="maze2d-large-v1"

# single
task="scripts/single/plan.py"
config="config.single.plan_diff"
plan_task="single"


# hier
# task="scripts/hier/plan.py"
# config='config.hier.plan_diff'
# plan_task="hier"

for cond in "True" "False"
do
    for idx in {0..150}
    do
        # Determine which GPU to assign based on the job index
        gpu_idx=$((current_jobs % NUM_GPUS))
        gpu="${GPU_IDS[$gpu_idx]}"

        python $task --dataset $dataset --config $config \
        --suffix "eval_$idx" --prefix plans/$plan_task --conditional $cond --device "$gpu" \
        &

        # Increment the current_jobs counter
        current_jobs=$((current_jobs + 1))

        # If the maximum number of parallel jobs is reached, wait for all of them to finish
        if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
            wait
            current_jobs=0
        fi
    done
done

# Wait for any remaining jobs to complete
wait

echo "Done"
