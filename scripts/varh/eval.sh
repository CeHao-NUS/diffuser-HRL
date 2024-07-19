
MAX_JOBS=5
current_jobs=0

for idx in {0..150}
do
    python scripts/varh/plan_maze2d.py --config config.varh.maze2d --dataset maze2d-umaze-v1 \
    --suffix "single_$idx" --min_horizon 16 &

    # Increment the current_jobs counter
    current_jobs=$((current_jobs + 1))

    # If the maximum number of parallel jobs is reached, wait for all of them to finish
    if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
        wait
        current_jobs=0
    fi
done

MAX_JOBS=5
current_jobs=0

for idx in {0..150}
do
    python scripts/varh/plan_maze2d.py --config config.varh.maze2d --dataset maze2d-umaze-v1 \
    --suffix "single_$idx" --min_horizon 64 &

    # Increment the current_jobs counter
    current_jobs=$((current_jobs + 1))

    # If the maximum number of parallel jobs is reached, wait for all of them to finish
    if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
        wait
        current_jobs=0
    fi
done