for idx in {0..150}
do
    python scripts/varh/plan_maze2d.py --config config.varh.maze2d --dataset maze2d-umaze-v1 \
    --suffix "single_$idx"
done