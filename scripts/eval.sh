# for idx in {0..150}
# do
#     python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-umaze-v1 \
#     --suffix "single_$idx"
# done

# for idx in {0..150}
# do
#     python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-medium-v1 \
#     --suffix "single_$idx"
# done


# for idx in {0..150}
# do
#     python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 \
#     --suffix "single_$idx"
# done


# ========================== multi ==========================
# for idx in {0..150}
# do
#     python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-umaze-v1 \
#     --suffix "multi_$idx" --conditional True
# done

# for idx in {0..150}
# do
#     python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-medium-v1 \
#     --suffix "multi_$idx" --conditional True
# done

for idx in {0..150}
do
    python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 \
    --suffix "multi_$idx" --conditional True
done