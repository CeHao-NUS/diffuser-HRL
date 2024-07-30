
python scripts/mazevarh/plan_guided.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1,1.5)" --target "(1, 1)" --suffix "near1.5" --prefix plans/testvarh_guided &

python scripts/mazevarh/plan_guided.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1,1.1)" --target "(1, 1)" --suffix "near1.1" --prefix plans/testvarh_guided &

python scripts/mazevarh/plan_guided.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(3,1)" --target "(1, 1)" --suffix "far1" --prefix plans/testvarh_guided &

python scripts/mazevarh/plan_guided.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(3,2)" --target "(1, 1)" --suffix "far2" --prefix plans/testvarh_guided &

python scripts/mazevarh/plan_guided.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(3,1.5)" --target "(1, 1)" --suffix "far1.5" --prefix plans/testvarh_guided &

python scripts/mazevarh/plan_guided.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(3,3)" --target "(1, 1)" --suffix "turn1" --prefix plans/testvarh_guided &

python scripts/mazevarh/plan_guided.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(2,3)" --target "(1, 1)" --suffix "turn0.5" --prefix plans/testvarh_guided &

python scripts/mazevarh/plan_guided.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1,3)" --target "(1, 1)" --suffix "line" --prefix plans/testvarh_guided &

python scripts/mazevarh/plan_guided.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1,2)" --target "(1, 1)" --suffix "near2" --prefix plans/testvarh_guided &

wait


# MAX_JOBS=15
# current_jobs=0

# for idx in {0..20}
# do
#     python scripts/mazevarh/plan_guided.py  --dataset maze2d-umaze-v1 \
#     --conditional True --init_pose "(1,2)" --target "(1, 1)" \
#     --suffix "(1, 2)_$idx" --prefix plans/testvarh_guided &

#     # Increment the current_jobs counter
#     current_jobs=$((current_jobs + 1))

#     # If the maximum number of parallel jobs is reached, wait for all of them to finish
#     if [ "$current_jobs" -ge "$MAX_JOBS" ]; then
#         wait
#         current_jobs=0
#     fi
# done