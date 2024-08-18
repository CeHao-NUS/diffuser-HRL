
# --dataset maze2d-umaze-v1
# --dataset maze2d-medium-v1
# --dataset maze2d-large-v1


# p-1_0: LL diffuser
python scripts/maze_stitch/train/train_diffuser.py --config 'config.stitch.train.train_LL_fixh'
python scripts/maze_stitch/plan/plan_1.py --config 'config.stitch.plan.plan_1_0'

# p-1_1: LL_varh
python scripts/maze_stitch/train/train_diffuser.py --config 'config.stitch.train.train_LL_varh'
python scripts/maze_stitch/plan/plan_1.py --config 'config.stitch.plan.plan_1_1'

# p-1_2: LL_value_varh
python scripts/maze_stitch/train/train_diffuser.py --config 'config.stitch.train.train_LL_varh'
python scripts/maze_stitch/train/train_value.py --config 'config.stitch.train.train_LL_varh'
python scripts/maze_stitch/plan/plan_1.py --config 'config.stitch.plan.plan_1_2'


# p-2_0: LL diffuer stitch
python scripts/maze_stitch/train/train_diffuser.py --config 'config.stitch.train.train_LL_fixh' --horizon 32 --n_diffusion_steps 32
python scripts/maze_stitch/plan/plan_2.py --config 'config.stitch.plan.plan_2_0' --seg_length 3



# p-3_0: HL_fix + LL_fix
python scripts/maze_stitch/train/train_diffuser.py --config 'config.stitch.train.train_LL_fixh' --horizon 16 --n_diffusion_steps 32
python scripts/maze_stitch/train/train_diffuser.py --config 'config.stitch.train.train_HL_fixh' --horizon 192 --n_diffusion_steps 32 --downsample 16
python scripts/maze_stitch/plan/plan_3.py --config 'config.stitch.plan.plan_3_0' --seg_length 11 \
 --LL_horizon 16 --HL_horizon 192 --n_diffusion_steps 32 --downsample 16

# p-3_1: HL_varh(guide) + LL_fix
python scripts/maze_stitch/train/train_diffuser.py --config 'config.stitch.train.train_HL_varh' \
 --horizon 192 --n_diffusion_steps 32 --downsample 16 --min_horizon 1

python scripts/maze_stitch/train/train_value.py  --config 'config.stitch.train.train_HL_varh' \
 --horizon 192 --n_diffusion_steps 32 --downsample 16 --min_horizon 1

python scripts/maze_stitch/plan/plan_3.py --config 'config.stitch.plan.plan_3_1' --seg_length 11 \
 --LL_horizon 16 --HL_horizon 192 --n_diffusion_steps 32 --downsample 16  --HL_min_horizon 1


# p-3_2: HL_fix + LL_varh(guide)
python scripts/maze_stitch/train/train_diffuser.py --config 'config.stitch.train.train_LL_varh' \
 --horizon 16 --n_diffusion_steps 32 --min_horizon 1

python scripts/maze_stitch/train/train_value.py  --config 'config.stitch.train.train_LL_varh' \
 --horizon 16 --n_diffusion_steps 32 --min_horizon 1

python scripts/maze_stitch/plan/plan_3.py --config 'config.stitch.plan.plan_3_2' --seg_length 11 \
 --LL_horizon 16 --HL_horizon 192 --n_diffusion_steps 32 --downsample 16 --LL_min_horizon 1

# p-3_3: HL_varh(guide) + LL_varh(guide)
python scripts/maze_stitch/plan/plan_3.py --config 'config.stitch.plan.plan_3_3' --seg_length 11 \
 --LL_horizon 16 --HL_horizon 192 --n_diffusion_steps 32 --downsample 16 --LL_min_horizon 1 --HL_min_horizon 1
