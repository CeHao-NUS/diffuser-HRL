MAX_JOBS=15
current_jobs=0

# bash scripts/maze_stitch/run/eval.sh 

# dataset="maze2d-umaze-v1"
# dataset="maze2d-medium-v1"
dataset="maze2d-large-v1"

plan_task="plan_2_0"

for idx in {0..150}
do
    python scripts/maze_stitch/plan/$plan_task.py --dataset $dataset \
    --suffix "eval_$idx" --prefix 'plans/$plan_task' &

    # Increment the current_jobs counter
    current_jobs=$((current_jobs + 1))

    # If the maximum number of parallel jobs is reached, wait for all of them to finish
    if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
        wait
        current_jobs=0
    fi
done


for idx in {0..150}
do
    python scripts/maze_stitch/plan/$plan_task.py --dataset $dataset \
    --suffix "eval_$idx" --prefix 'plans/$plan_task' --conditional True &

    # Increment the current_jobs counter
    current_jobs=$((current_jobs + 1))

    # If the maximum number of parallel jobs is reached, wait for all of them to finish
    if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
        wait
        current_jobs=0
    fi
done

echo "Done"

