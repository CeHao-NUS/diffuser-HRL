# bash scripts/maze_stitch/run/eval_for_p_1.sh 

# p-1_1
for init_pose in "(1,1.5)" "(1,1.1)" "(3,1)" "(3,2)" "(3,1.5)" "(3,3)" "(2,3)" "(1,3)" "(1,2)"
do
    python scripts/maze_stitch/plan/plan_1.py --config 'config.stitch.plan.plan_1_1' \
    --conditional True --init_pose $init_pose --target "(1, 1)" --suffix $init_pose &
done

wait

# p-1_2 
# for init_pose in "(1,1.5)" "(1,1.1)" "(3,1)" "(3,2)" "(3,1.5)" "(3,3)" "(2,3)" "(1,3)" "(1,2)"
# do
#     python scripts/maze_stitch/plan/plan_1.py --config 'config.stitch.plan.plan_1_2' \
#     --conditional True --init_pose $init_pose --target "(1, 1)" --suffix $init_pose &
# done

# wait