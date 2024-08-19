# bash scripts/maze_stitch/run/exp_for_p_3.sh 

# p-3_0: HL_fix + LL_fix
for init_pose in "(1,1.5)" "(1,1.1)" "(3,1)" "(3,2)" "(3,1.5)" "(3,3)" "(2,3)" "(1,3)" "(1,2)"
do
    python scripts/maze_stitch/plan/plan_3.py --config 'config.stitch.plan.plan_3_0' --seg_length 11 \
    --LL_horizon 16 --HL_horizon 192 --n_diffusion_steps 32 --downsample 16 \
    --conditional True --init_pose $init_pose --target "(1, 1)" --suffix $init_pose &
done

wait

# p-3_1: HL_varh(guide) + LL_fix
# for init_pose in "(1,1.5)" "(1,1.1)" "(3,1)" "(3,2)" "(3,1.5)" "(3,3)" "(2,3)" "(1,3)" "(1,2)"
# do
#     python scripts/maze_stitch/plan/plan_3.py --config 'config.stitch.plan.plan_3_1' --seg_length 11 \
#     --LL_horizon 16 --HL_horizon 192 --n_diffusion_steps 32 --downsample 16  --HL_min_horizon 1 \
#     --conditional True --init_pose $init_pose --target "(1, 1)" --suffix $init_pose &
# done

# wait


# p-3_2: HL_fix + LL_varh(guide)
# for init_pose in "(1,1.5)" "(1,1.1)" "(3,1)" "(3,2)" "(3,1.5)" "(3,3)" "(2,3)" "(1,3)" "(1,2)"
# do
#     python scripts/maze_stitch/plan/plan_3.py --config 'config.stitch.plan.plan_3_2' --seg_length 11 \
#     --LL_horizon 16 --HL_horizon 192 --n_diffusion_steps 32 --downsample 16 --LL_min_horizon 1 \
#     --conditional True --init_pose $init_pose --target "(1, 1)" --suffix $init_pose &
# done

# wait

# p-3_3: HL_varh(guide) + LL_varh(guide)
# for init_pose in "(1,1.5)" "(1,1.1)" "(3,1)" "(3,2)" "(3,1.5)" "(3,3)" "(2,3)" "(1,3)" "(1,2)"
# do
#     python scripts/maze_stitch/plan/plan_3.py --config 'config.stitch.plan.plan_3_3' --seg_length 11 \
#     --LL_horizon 16 --HL_horizon 192 --n_diffusion_steps 32 --downsample 16 --LL_min_horizon 1 --HL_min_horizon 1 \
#     --conditional True --init_pose $init_pose --target "(1, 1)" --suffix $init_pose &
# done

# wait