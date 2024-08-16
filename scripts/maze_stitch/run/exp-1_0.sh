MAX_JOBS=15
current_jobs=0

for idx in {1..4}
do

    python scripts/maze_stitch/plan/plan-1_0.py --dataset maze2d-umaze-v1 \
    --conditional True --init_pose "(1,1.5)" --target "(1, 1)" --suffix "near1.5" --prefix plans/test_stitch --seg_length $idx &

    python scripts/maze_stitch/plan/plan-1_0.py --dataset maze2d-umaze-v1 \
    --conditional True --init_pose "(1,1.5)" --target "(1, 1)" --suffix "near1.5" --prefix plans/test_stitch --seg_length $idx &

    python scripts/maze_stitch/plan/plan-1_0.py --dataset maze2d-umaze-v1 \
    --conditional True --init_pose "(1,1.1)" --target "(1, 1)" --suffix "near1.1" --prefix plans/test_stitch --seg_length $idx &

    python scripts/maze_stitch/plan/plan-1_0.py --dataset maze2d-umaze-v1 \
    --conditional True --init_pose "(3,1)" --target "(1, 1)" --suffix "far1" --prefix plans/test_stitch --seg_length $idx  &

    python scripts/maze_stitch/plan/plan-1_0.py --dataset maze2d-umaze-v1 \
    --conditional True --init_pose "(3,2)" --target "(1, 1)" --suffix "far2" --prefix plans/test_stitch --seg_length $idx  &

    python scripts/maze_stitch/plan/plan-1_0.py --dataset maze2d-umaze-v1 \
    --conditional True --init_pose "(3,1.5)" --target "(1, 1)" --suffix "far1.5" --prefix plans/test_stitch --seg_length $idx  &

    python scripts/maze_stitch/plan/plan-1_0.py --dataset maze2d-umaze-v1 \
    --conditional True --init_pose "(3,3)" --target "(1, 1)" --suffix "turn1" --prefix plans/test_stitch --seg_length $idx  &

    python scripts/maze_stitch/plan/plan-1_0.py --dataset maze2d-umaze-v1 \
    --conditional True --init_pose "(2,3)" --target "(1, 1)" --suffix "turn0.5" --prefix plans/test_stitch --seg_length $idx  &

    python scripts/maze_stitch/plan/plan-1_0.py --dataset maze2d-umaze-v1 \
    --conditional True --init_pose "(1,3)" --target "(1, 1)" --suffix "line" --prefix plans/test_stitch --seg_length $idx  &

    python scripts/maze_stitch/plan/plan-1_0.py --dataset maze2d-umaze-v1 \
    --conditional True --init_pose "(1,2)" --target "(1, 1)" --suffix "near2" --prefix plans/test_stitch --seg_length $idx  &

    wait
done