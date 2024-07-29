
python scripts/mazerope/plan_rope.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1,1.5)" --target "(1, 1)" --suffix "near1.5" --prefix plans/testrope &

python scripts/mazerope/plan_rope.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1,1.1)" --target "(1, 1)" --suffix "near1.1" --prefix plans/testrope &

python scripts/mazerope/plan_rope.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(3,1)" --target "(1, 1)" --suffix "far1" --prefix plans/testrope &

python scripts/mazerope/plan_rope.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(3,2)" --target "(1, 1)" --suffix "far2" --prefix plans/testrope &

python scripts/mazerope/plan_rope.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(3,1.5)" --target "(1, 1)" --suffix "far1.5" --prefix plans/testrope &

python scripts/mazerope/plan_rope.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(3,3)" --target "(1, 1)" --suffix "turn1" --prefix plans/testrope &

python scripts/mazerope/plan_rope.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(2,3)" --target "(1, 1)" --suffix "turn0.5" --prefix plans/testrope &

python scripts/mazerope/plan_rope.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1,3)" --target "(1, 1)" --suffix "line" --prefix plans/testrope &

python scripts/mazerope/plan_rope.py --dataset maze2d-umaze-v1 \
--conditional True --init_pose "(1,2)" --target "(1, 1)" --suffix "near2" --prefix plans/testrope &

wait

