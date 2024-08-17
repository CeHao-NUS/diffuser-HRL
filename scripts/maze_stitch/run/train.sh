

# method 1:

python scripts/maze_stitch/train/train_LL.py 
python scripts/maze_stitch/plan/plan-1_0.py

python scripts/maze_stitch/train/train_HL.py 


python scripts/maze_stitch/train/train_LL.py  --dataset maze2d-medium-v1
python scripts/maze_stitch/train/train_LL.py  --dataset maze2d-large-v1

python scripts/maze_stitch/train/train_HL.py --dataset maze2d-medium-v1
python scripts/maze_stitch/train/train_HL.py --dataset maze2d-large-v1





maze2d-umaze-v1
maze2d-medium-v1
maze2d-large-v1

python scripts/maze_stitch/train/train_LL_varh.py --dataset maze2d-umaze-v1
python scripts/maze_stitch/train/train_LL_value_varh.py  --dataset maze2d-umaze-v1

python scripts/maze_stitch/train/train_LL_varh.py --dataset maze2d-medium-v1
python scripts/maze_stitch/train/train_LL_value_varh.py  --dataset maze2d-medium-v1

python scripts/maze_stitch/train/train_LL_varh.py --dataset maze2d-large-v1
python scripts/maze_stitch/train/train_LL_value_varh.py  --dataset maze2d-large-v1


 
# train new method

1. LL fix 16
2. HL fix 192//16
3. LL varh 16
4. LL value varh 16
5. HL varh 192//16
6. HL value varh 192//16