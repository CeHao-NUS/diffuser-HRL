
# MAX_JOBS=5
# current_jobs=0

# for idx in {0..150}
# do
#     python scripts/hier/plan_maze2d.py --dataset maze2d-umaze-v1 --suffix "single_$idx" &

#     # Increment the current_jobs counter
#     current_jobs=$((current_jobs + 1))

#     # If the maximum number of parallel jobs is reached, wait for all of them to finish
#     if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
#         wait
#         current_jobs=0
#     fi
# done

MAX_JOBS=5
current_jobs=0

for idx in {0..150}
do
    python scripts/hier/plan_maze2d.py --dataset maze2d-medium-v1 --suffix "single_$idx" &

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
    python scripts/hier/plan_maze2d.py --dataset maze2d-large-v1 --suffix "single_$idx" &

    # Increment the current_jobs counter
    current_jobs=$((current_jobs + 1))

    if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
        wait
        current_jobs=0
    fi
done

# ========================== multi ==========================
MAX_JOBS=5
current_jobs=0

for idx in {0..150}
do
    python scripts/hier/plan_maze2d.py --dataset maze2d-umaze-v1 --suffix "multi_$idx" --conditional True &

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
    python scripts/hier/plan_maze2d.py --dataset maze2d-medium-v1 --suffix "multi_$idx" --conditional True &

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
    python scripts/hier/plan_maze2d.py --dataset maze2d-large-v1 --suffix "multi_$idx" --conditional True &

    # Increment the current_jobs counter
    current_jobs=$((current_jobs + 1))

    # If the maximum number of parallel jobs is reached, wait for all of them to finish
    if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
        wait
        current_jobs=0
    fi
done