# bash scripts/eval/exp_multi_test.sh

MAX_JOBS=5
GPU_IDS=("cuda:0" "cuda:1" "cuda:2" "cuda:3" "cuda:4" "cuda:5" "cuda:6" "cuda:7")
NUM_GPUS=${#GPU_IDS[@]} # Number of GPUs (8 in this case)

# Array to track the number of running jobs on each GPU
declare -a gpu_jobs
for ((i=0; i<NUM_GPUS; i++)); do
    gpu_jobs[$i]=0
done

dataset="maze2d-umaze-v1"
# dataset="maze2d-medium-v1"
# dataset="maze2d-large-v1"

# hier
task="scripts/hier/plan.py"
config='config.hier.plan_diff'
plan_task="hier"

for cond in "True" "False"
do
    for idx in {0..150}
    do
        while true
        do
            # Loop through GPUs and assign the job to the first GPU with available capacity
            for gpu_idx in "${!GPU_IDS[@]}"
            do
                if [ "${gpu_jobs[$gpu_idx]}" -lt "$MAX_JOBS" ]; then
                    gpu="${GPU_IDS[$gpu_idx]}"
                    
                    # Launch the job and keep track of its PID
                    python $task --dataset $dataset --config $config \
                    --suffix "eval_$idx" --prefix plans/$plan_task --conditional $cond --device "$gpu" \
                    &
                    pid=$!

                    # Increment the number of jobs on this GPU
                    gpu_jobs[$gpu_idx]=$((gpu_jobs[$gpu_idx] + 1))

                    # Store the PID to ensure we can track the job status
                    gpu_pids[$gpu_idx]+=" $pid"

                    # Break the loop and proceed to the next job
                    break
                fi
            done

            # Check if we need to wait for some jobs to finish before assigning new ones
            jobs_running=0
            for jobs in "${gpu_jobs[@]}"; do
                jobs_running=$((jobs_running + jobs))
            done

            # If there are less than NUM_GPUS * MAX_JOBS jobs running, continue with the next job
            if [ "$jobs_running" -lt "$((NUM_GPUS * MAX_JOBS))" ]; then
                break
            fi

            # Wait for some jobs to complete before checking again
            wait -n

            # Decrement job count for any finished jobs
            for ((i=0; i<NUM_GPUS; i++)); do
                for pid in ${gpu_pids[$i]}; do
                    if ! kill -0 $pid 2>/dev/null; then
                        gpu_jobs[$i]=$((gpu_jobs[$i] - 1))
                        gpu_pids[$i]=$(echo ${gpu_pids[$i]} | sed "s/\b$pid\b//g")
                    fi
                done
            done
        done
    done
done

# Wait for any remaining jobs to complete
wait

echo "Done"
