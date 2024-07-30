# maze2d, guided, single/multi


# 1. plan 2d

# MAX_JOBS=15
MAX_JOBS=10

# ========================== single ==========================
# current_jobs=0

# for idx in {0..150}
# do
#     python scripts/mazevarh/plan_maze2d.py  --dataset maze2d-umaze-v1 \
#     --suffix "single_$idx" --prefix 'plans/mh_No' &

#     # Increment the current_jobs counter
#     current_jobs=$((current_jobs + 1))

#     # If the maximum number of parallel jobs is reached, wait for all of them to finish
#     if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
#         wait
#         current_jobs=0
#     fi
# done

# current_jobs=0

# for idx in {0..150}
# do
#     python scripts/mazevarh/plan_maze2d.py  --dataset maze2d-medium-v1 \
#     --suffix "single_$idx" --prefix 'plans/mh_No' &

#     # Increment the current_jobs counter
#     current_jobs=$((current_jobs + 1))

#     # If the maximum number of parallel jobs is reached, wait for all of them to finish
#     if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
#         wait
#         current_jobs=0
#     fi
# done

current_jobs=0

for idx in {0..150}
do
    python scripts/mazevarh/plan_maze2d.py  --dataset maze2d-large-v1 \
    --suffix "single_$idx" --prefix 'plans/mh_No' &

    # Increment the current_jobs counter
    current_jobs=$((current_jobs + 1))

    # If the maximum number of parallel jobs is reached, wait for all of them to finish
    if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
        wait
        current_jobs=0
    fi
done

# ========================== multi ==========================

# current_jobs=0

# for idx in {0..150}
# do
#     python scripts/mazevarh/plan_maze2d.py  --dataset maze2d-umaze-v1 \
#     --suffix "multi_$idx" --conditional True --prefix 'plans/mh_No' &

#     # Increment the current_jobs counter
#     current_jobs=$((current_jobs + 1))

#     # If the maximum number of parallel jobs is reached, wait for all of them to finish
#     if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
#         wait
#         current_jobs=0
#     fi
# done

# current_jobs=0

# for idx in {0..150}
# do
#     python scripts/mazevarh/plan_maze2d.py  --dataset maze2d-medium-v1 \
#     --suffix "multi_$idx" --conditional True --prefix 'plans/mh_No' &

#     # Increment the current_jobs counter
#     current_jobs=$((current_jobs + 1))

#     # If the maximum number of parallel jobs is reached, wait for all of them to finish
#     if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
#         wait
#         current_jobs=0
#     fi
# done

# current_jobs=0

for idx in {0..150}
do
    python scripts/mazevarh/plan_maze2d.py  --dataset maze2d-large-v1 \
    --suffix "multi_$idx" --conditional True --prefix 'plans/mh_No' &

    # Increment the current_jobs counter
    current_jobs=$((current_jobs + 1))

    # If the maximum number of parallel jobs is reached, wait for all of them to finish
    if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
        wait
        current_jobs=0
    fi
done